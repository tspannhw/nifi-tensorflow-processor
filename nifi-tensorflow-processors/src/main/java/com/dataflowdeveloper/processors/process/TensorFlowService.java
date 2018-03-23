package com.dataflowdeveloper.processors.process;

//see:   https://raw.githubusercontent.com/tensorflow/tensorflow/r1.6/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.types.UInt8;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * copied from TensorFlow example
 * 
 * @author tspann
 *
 */
public class TensorFlowService {

	public static final String IMAGENET_COMP_GRAPH_LABEL_STRINGS_TXT = "imagenet_comp_graph_label_strings.txt";
	public static final String TENSORFLOW_INCEPTION_GRAPH_PB = "tensorflow_inception_graph.pb";

	private final Logger logger = LoggerFactory.getLogger(TensorFlowService.class);
	private Map<Path, byte[]> modelCache = new HashMap<Path, byte[]>();
	private Map<Path, List<String>> labelCache = new HashMap<Path, List<String>>();

	/**
	 * cache labels
	 * 
	 * @param path
	 * @return
	 */
	private List<String> getOrCreateLabels(Path path) {
		if (labelCache.containsKey(path)) {
			return labelCache.get(path);
		}
		labelCache.put(path, readAllLinesOrExit(path));
		return labelCache.get(path);
	}

	/**
	 * cache path bytes
	 * 
	 * @param path
	 * @return
	 */
	private byte[] getOrCreate(Path path) {
		if (modelCache.containsKey(path)) {
			return modelCache.get(path);
		}
		byte[] graphDef = readAllBytesOrExit(path);
		modelCache.put(path, graphDef);
		return modelCache.get(path);
	}

	/**
	 * get inception
	 * List<Entry<String, String>>
	 * @param imageBytes
	 * @param modelDir
	 * @return map
	 */
	public List<InceptionResult> getInception(byte[] imageBytes, String modelDir) {

		byte[] graphDef = getOrCreate(Paths.get(modelDir, TENSORFLOW_INCEPTION_GRAPH_PB));
		List<String> labels = getOrCreateLabels(Paths.get(modelDir, IMAGENET_COMP_GRAPH_LABEL_STRINGS_TXT));

		logger.debug(String.format("getInception: %d bytes %s",
				new Object[] { imageBytes.length, Paths.get(modelDir, TENSORFLOW_INCEPTION_GRAPH_PB) }));

		List<InceptionResult> results = new ArrayList<>();
		
		try (Tensor<Float> image = constructAndExecuteGraphToNormalizeImage(imageBytes)) {
			float[] labelProbabilities = executeInceptionGraph(graphDef, image);
			float[] large = new float[labelProbabilities.length];
			float[] labelProbabilities2 = labelProbabilities.clone();
		    float max = 0;
		    int index;
		    String label = null;
		    
		    for (int j = 0; j < 5; j++) {
		        max = labelProbabilities2[0];
		        index = 0;
		        for (int i = 1; i < labelProbabilities2.length; i++) {
		            if (max < labelProbabilities2[i]) {
		                max = labelProbabilities2[i];
		                label = labels.get(i);
		                index = i;
		            } 
		        }
		        large[j] = max;
		        labelProbabilities2[index] = Integer.MIN_VALUE;
		        results.add(new InceptionResult(label, String.format("%.2f%%", ( large[j] )* 100f), j));
		    }
		    
		    return results;

		} catch (Exception e) {
			logger.error("Failed in tensorflow", e);
			throw (e);
		}
	}

	/**
	 * 
	 * @param imageBytes
	 * @return
	 */
	private static Tensor<Float> constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
		try (Graph g = new Graph()) {
			GraphBuilder b = new GraphBuilder(g);
			// Some constants specific to the pre-trained model at:
			// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
			//
			// - The model was trained with images scaled to 224x224 pixels.
			// - The colors, represented as R, G, B in 1-byte each were converted to
			// float using (value - Mean)/Scale.
			final int H = 224;
			final int W = 224;
			final float mean = 117f;
			final float scale = 1f;

			// Since the graph is being constructed once per execution here, we can use a
			// constant for the
			// input image. If the graph were to be re-used for multiple input images, a
			// placeholder would
			// have been more appropriate.
			final Output<String> input = b.constant("input", imageBytes);
			final Output<Float> output = b
					.div(b.sub(
							b.resizeBilinear(b.expandDims(b.cast(b.decodeJpeg(input, 3), Float.class),
									b.constant("make_batch", 0)), b.constant("size", new int[] { H, W })),
							b.constant("mean", mean)), b.constant("scale", scale));
			try (Session s = new Session(g)) {
				return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
			}
		}
	}

	private static float[] executeInceptionGraph(byte[] graphDef, Tensor<Float> image) {
		try (Graph g = new Graph()) {
			g.importGraphDef(graphDef);
			try (Session s = new Session(g);
					Tensor<Float> result = s.runner().feed("input", image).fetch("output").run().get(0)
							.expect(Float.class)) {
				final long[] rshape = result.shape();
				if (result.numDimensions() != 2 || rshape[0] != 1) {
					throw new RuntimeException(String.format(
							"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
							Arrays.toString(rshape)));
				}
				int nlabels = (int) rshape[1];
				return result.copyTo(new float[1][nlabels])[0];
			}
		}
	}

	private static byte[] readAllBytesOrExit(Path path) {
		try {
			return Files.readAllBytes(path);
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			System.exit(1);
		}
		return null;
	}

	private static List<String> readAllLinesOrExit(Path path) {
		try {
			return Files.readAllLines(path, Charset.forName("UTF-8"));
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			System.exit(0);
		}
		return null;
	}

	// In the fullness of time, equivalents of the methods of this class should be
	// auto-generated from
	// the OpDefs linked into libtensorflow_jni.so. That would match what is done in
	// other languages
	// like Python, C++ and Go.
	static class GraphBuilder {
		GraphBuilder(Graph g) {
			this.g = g;
		}

		Output<Float> div(Output<Float> x, Output<Float> y) {
			return binaryOp("Div", x, y);
		}

		<T> Output<T> sub(Output<T> x, Output<T> y) {
			return binaryOp("Sub", x, y);
		}

		<T> Output<Float> resizeBilinear(Output<T> images, Output<Integer> size) {
			return binaryOp3("ResizeBilinear", images, size);
		}

		<T> Output<T> expandDims(Output<T> input, Output<Integer> dim) {
			return binaryOp3("ExpandDims", input, dim);
		}

		<T, U> Output<U> cast(Output<T> value, Class<U> type) {
			DataType dtype = DataType.fromClass(type);
			return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().<U>output(0);
		}

		Output<UInt8> decodeJpeg(Output<String> contents, long channels) {
			return g.opBuilder("DecodeJpeg", "DecodeJpeg").addInput(contents).setAttr("channels", channels).build()
					.<UInt8>output(0);
		}

		<T> Output<T> constant(String name, Object value, Class<T> type) {
			try (Tensor<T> t = Tensor.<T>create(value, type)) {
				return g.opBuilder("Const", name).setAttr("dtype", DataType.fromClass(type)).setAttr("value", t).build()
						.<T>output(0);
			}
		}

		Output<String> constant(String name, byte[] value) {
			return this.constant(name, value, String.class);
		}

		Output<Integer> constant(String name, int value) {
			return this.constant(name, value, Integer.class);
		}

		Output<Integer> constant(String name, int[] value) {
			return this.constant(name, value, Integer.class);
		}

		Output<Float> constant(String name, float value) {
			return this.constant(name, value, Float.class);
		}

		private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
		}

		private <T, U, V> Output<T> binaryOp3(String type, Output<U> in1, Output<V> in2) {
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().<T>output(0);
		}

		private Graph g;
	}
}