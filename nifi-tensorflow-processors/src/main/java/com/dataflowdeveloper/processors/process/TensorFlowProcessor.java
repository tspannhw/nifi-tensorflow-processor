/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.dataflowdeveloper.processors.process;

import java.io.IOException;
import java.io.InputStream;

// see:   https://raw.githubusercontent.com/tensorflow/tensorflow/r1.2/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.IOUtils;
import org.apache.nifi.annotation.behavior.ReadsAttribute;
import org.apache.nifi.annotation.behavior.ReadsAttributes;
import org.apache.nifi.annotation.behavior.WritesAttribute;
import org.apache.nifi.annotation.behavior.WritesAttributes;
import org.apache.nifi.annotation.documentation.CapabilityDescription;
import org.apache.nifi.annotation.documentation.SeeAlso;
import org.apache.nifi.annotation.documentation.Tags;
import org.apache.nifi.annotation.lifecycle.OnScheduled;
import org.apache.nifi.components.PropertyDescriptor;
import org.apache.nifi.flowfile.FlowFile;
import org.apache.nifi.processor.AbstractProcessor;
import org.apache.nifi.processor.ProcessContext;
import org.apache.nifi.processor.ProcessSession;
import org.apache.nifi.processor.ProcessorInitializationContext;
import org.apache.nifi.processor.Relationship;
import org.apache.nifi.processor.exception.ProcessException;
import org.apache.nifi.processor.io.InputStreamCallback;
import org.apache.nifi.processor.util.StandardValidators;

@Tags({ "tensorflowprocessor" })
@CapabilityDescription("Run TensorFlow Image Recognition")
@SeeAlso({})
@ReadsAttributes({ @ReadsAttribute(attribute = "", description = "") })
@WritesAttributes({ @WritesAttribute(attribute = "", description = "") })
/**
 * 
 * @author tspann
 *
 */
public class TensorFlowProcessor extends AbstractProcessor {

	public static final String ATTRIBUTE_OUTPUT_NAME = "probabilities";
	public static final String MODEL_DIR_NAME = "modeldir";
	public static final String PROPERTY_NAME_EXTRA = "Extra Resources";

	public static final PropertyDescriptor MODEL_DIR = new PropertyDescriptor.Builder().name(MODEL_DIR_NAME)
			.description("Model Directory").required(true).expressionLanguageSupported(true)
			.addValidator(StandardValidators.NON_EMPTY_VALIDATOR).build();

	public static final Relationship REL_SUCCESS = new Relationship.Builder().name("success")
			.description("Successfully determined image.").build();
	public static final Relationship REL_UNMATCHED = new Relationship.Builder().name("unmatched")
			.description("Found no matches in image").build();

	public static final Relationship REL_FAILURE = new Relationship.Builder().name("failure")
			.description("Failed to determine image.").build();

	private List<PropertyDescriptor> descriptors;

	private Set<Relationship> relationships;

	private TensorFlowService service;

	@Override
	protected void init(final ProcessorInitializationContext context) {
		final List<PropertyDescriptor> descriptors = new ArrayList<PropertyDescriptor>();
		descriptors.add(MODEL_DIR);
		this.descriptors = Collections.unmodifiableList(descriptors);

		final Set<Relationship> relationships = new HashSet<Relationship>();
		relationships.add(REL_SUCCESS);
		relationships.add(REL_FAILURE);
		this.relationships = Collections.unmodifiableSet(relationships);
	}

	@Override
	public Set<Relationship> getRelationships() {
		return this.relationships;
	}

	@Override
	public final List<PropertyDescriptor> getSupportedPropertyDescriptors() {
		return descriptors;
	}

	@OnScheduled
	public void onScheduled(final ProcessContext context) {
		return;
	}

	@Override
	public void onTrigger(final ProcessContext context, final ProcessSession session) throws ProcessException {
		FlowFile flowFile = session.get();
		if (flowFile == null) {
			flowFile = session.create();
		}
		try {
			flowFile.getAttributes();

			String modelDir = flowFile.getAttribute(MODEL_DIR_NAME);
			if (modelDir == null) {
				modelDir = context.getProperty(MODEL_DIR_NAME).evaluateAttributeExpressions(flowFile).getValue();
			}
			if (modelDir == null) {
				modelDir = "/models";
			}

			service = new TensorFlowService();
			// read all bytes of the flowfile (tensor requires whole image)
			InputStream is = session.read(flowFile);
			try {
				byte[] byteArray = IOUtils.toByteArray(is);
				String value = service.getInception(byteArray, modelDir);

				if (value == null) {
					session.transfer(flowFile, REL_UNMATCHED);
				} else {
					flowFile = session.putAttribute(flowFile, ATTRIBUTE_OUTPUT_NAME, value);
					session.transfer(flowFile, REL_SUCCESS);
				}
			} catch(Exception e) {
				throw new ProcessException(e);
			} finally {
				is.close();
			}

			session.commit();
		} catch (final Throwable t) {
			getLogger().error("Unable to process TensorFlow Processor file " + t.getLocalizedMessage());
			throw new ProcessException(t);
		}
	}
}
