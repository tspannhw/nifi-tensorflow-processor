package com.dataflowdeveloper.processors.process;

//see:   https://raw.githubusercontent.com/tensorflow/tensorflow/r1.2/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * copied from TensorFlow example
 * 
 * @author tspann
 *
 */
public class TensorFlowService {

	private final Logger logger = LoggerFactory.getLogger(TensorFlowService.class);

	private Map<Path, Graph> modelCache = new HashMap<Path, Graph>();
	private Map<Path, List<String>> labelCache = new HashMap<Path, List<String>>();

	public List<Entry<Float, String>> getInception(byte[] imageBytes, String modelDir) {
		logger.info(String.format("getInception: %d bytes %s", new Object[] { imageBytes.length, Paths.get(modelDir, "graph.pb") }));
		Graph g = getOrCreate(Paths.get(modelDir, "graph.pb"));
		try (Session s = new Session(g)) {
			List<String> labels = getOrCreateLabels(Paths.get(modelDir, "label.txt"));
			Tensor image = constructAndExecuteGraphToNormalizeImage(imageBytes);
			Tensor result = s.runner().feed("input", image).fetch("output").run().get(0);
			logger.debug("found results");

			final long[] rshape = result.shape();
			if (result.numDimensions() != 2 || rshape[0] != 1) {
				throw new RuntimeException(String.format(
						"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
						Arrays.toString(rshape)));
			}
			int nlabels = (int) rshape[1];

			logger.debug(String.format("number of labels %d, %d", new Object[] { labels.size(), nlabels }));
			int mLabeled = Math.min(labels.size(), nlabels);

			float[] labelProbabilities = result.copyTo(new float[1][nlabels])[0];

			HashMap<Float, String> results = new HashMap<Float, String>();
			for (int i = 0; i < mLabeled; i++) {
				results.put(labelProbabilities[i], labels.get(i));
			}

			return Collections.synchronizedList(
					results.entrySet().stream().sorted(Collections.reverseOrder(Map.Entry.comparingByKey())).limit(10)
							.collect(Collectors.toList()));
		} catch (Exception e) {
			logger.error("Failed in tensorflow", e);
			throw(e);
		}
	}

	private List<String> getOrCreateLabels(Path path) {
		if (labelCache.containsKey(path)) {
			return labelCache.get(path);
		}
		labelCache.put(path, readAllLinesOrExit(path));
		return labelCache.get(path);
	}

	private Graph getOrCreate(Path path) {
		if (modelCache.containsKey(path)) {
			return modelCache.get(path);
		}
		Graph g = new Graph();
		byte[] graphDef = readAllBytesOrExit(path);
		g.importGraphDef(graphDef);
		modelCache.put(path, g);
		return modelCache.get(path);
	}

	private static Tensor constructAndExecuteGraphToNormalizeImage(byte[] imageBytes) {
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
			final Output input = b.constant("input", imageBytes);
			final Output output = b
					.div(b.sub(
							b.resizeBilinear(b.expandDims(b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
									b.constant("make_batch", 0)), b.constant("size", new int[] { H, W })),
							b.constant("mean", mean)), b.constant("scale", scale));
			try (Session s = new Session(g)) {
				return s.runner().fetch(output.op().name()).run().get(0);
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

		Output div(Output x, Output y) {
			return binaryOp("Div", x, y);
		}

		Output sub(Output x, Output y) {
			return binaryOp("Sub", x, y);
		}

		Output resizeBilinear(Output images, Output size) {
			return binaryOp("ResizeBilinear", images, size);
		}

		Output expandDims(Output input, Output dim) {
			return binaryOp("ExpandDims", input, dim);
		}

		Output cast(Output value, DataType dtype) {
			return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
		}

		Output decodeJpeg(Output contents, long channels) {
			return g.opBuilder("DecodeJpeg", "DecodeJpeg").addInput(contents).setAttr("channels", channels).build()
					.output(0);
		}

		Output constant(String name, Object value) {
			try (Tensor t = Tensor.create(value)) {
				return g.opBuilder("Const", name).setAttr("dtype", t.dataType()).setAttr("value", t).build().output(0);
			}
		}

		private Output binaryOp(String type, Output in1, Output in2) {
			return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
		}

		private Graph g;
	}

}