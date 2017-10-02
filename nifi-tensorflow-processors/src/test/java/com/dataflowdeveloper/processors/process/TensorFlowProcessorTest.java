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

import static org.junit.Assert.assertNotNull;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.List;

import org.apache.nifi.components.PropertyDescriptor;
import org.apache.nifi.processor.util.StandardValidators;
import org.apache.nifi.util.MockFlowFile;
import org.apache.nifi.util.TestRunner;
import org.apache.nifi.util.TestRunners;
import org.junit.Before;
import org.junit.Test;

public class TensorFlowProcessorTest {

	private TestRunner testRunner;

	@Before
	public void init() {
		testRunner = TestRunners.newTestRunner(TensorFlowProcessor.class);
	}

	private String pathOfResource(String name) throws URISyntaxException {
		URL r = this.getClass().getClassLoader().getResource(name);
		URI uri = r.toURI();
		return Paths.get(uri).toAbsolutePath().getParent().toString();
	}

	@Test
	public void testProcessor() throws Exception {
		testRunner.setProperty(TensorFlowProcessor.MODEL_DIR, pathOfResource("models/graph.pb"));
		testRunner.enqueue(this.getClass().getClassLoader().getResourceAsStream("test.jpg"));

		runAndAssertHappy();
	}

	@Test
	public void testReruns() throws Exception {
		testRunner.setProperty(TensorFlowProcessor.MODEL_DIR, pathOfResource("models/graph.pb"));
		testRunner.enqueue(this.getClass().getClassLoader().getResourceAsStream("test.jpg"));
		testRunner.enqueue(this.getClass().getClassLoader().getResourceAsStream("test.jpg"));

		runAndAssertHappy();
	}

	private void runAndAssertHappy() {
		testRunner.setValidateExpressionUsage(false);
		testRunner.run();
		testRunner.assertValid();
		testRunner.assertAllFlowFilesTransferred(TensorFlowProcessor.REL_SUCCESS);
		List<MockFlowFile> successFiles = testRunner.getFlowFilesForRelationship(TensorFlowProcessor.REL_SUCCESS);

		for (MockFlowFile mockFile : successFiles) {
			System.out.println("Attribute: " + mockFile.getAttribute(TensorFlowProcessor.ATTRIBUTE_OUTPUT_NAME));
			assertNotNull(mockFile.getAttribute(TensorFlowProcessor.ATTRIBUTE_OUTPUT_NAME));
		}
	}
}
