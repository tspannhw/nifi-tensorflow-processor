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


        public static final String ATTRIBUTE_OUTPUT_NAME = "probabilities";
        public static final String ATTRIBUTE_INPUT_NAME = "imgpath";
    	public static final String ATTRIBUTE_INPUT_NAME2 = "modeldir";
        public static final String PROPERTY_NAME_EXTRA = "Extra Resources";

    public static final PropertyDescriptor MY_PROPERTY = new PropertyDescriptor
            .Builder().name(ATTRIBUTE_INPUT_NAME)
            .description("image")
            .required(true)
            .expressionLanguageSupported(true)
            .addValidator(StandardValidators.NON_EMPTY_VALIDATOR)
            .build();
    public static final PropertyDescriptor MY_PROPERTY2 = new PropertyDescriptor
            .Builder().name(ATTRIBUTE_INPUT_NAME2)
            .description("Model Directory")
            .required(true)
            .expressionLanguageSupported(true)
            .addValidator(StandardValidators.NON_EMPTY_VALIDATOR)
            .build();
    
	@Before
	public void init() {
		testRunner = TestRunners.newTestRunner(TensorFlowProcessor.class);
	}

	@Test
	public void testProcessor() {
		
		testRunner.setProperty(MY_PROPERTY, "/Volumes/Transcend/projects/nifi-tensorflow-processor/images/squirrelshirt.jpg");
		testRunner.setProperty(MY_PROPERTY2, "/Volumes/Transcend/projects/nifi-tensorflow-processor/models/");
		
		try {
			testRunner.enqueue(new FileInputStream(new File("src/test/resources/test.txt")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		testRunner.setValidateExpressionUsage(false);
		testRunner.run();
		testRunner.assertValid();
		List<MockFlowFile> successFiles = testRunner.getFlowFilesForRelationship(TensorFlowProcessor.REL_SUCCESS);

		for (MockFlowFile mockFile : successFiles) {
			try {
				System.out.println("FILE:" + new String(mockFile.toByteArray(), "UTF-8"));
				System.out.println("Attribute: " + mockFile.getAttribute(TensorFlowProcessor.ATTRIBUTE_OUTPUT_NAME));
				
				assertNotNull(  mockFile.getAttribute(TensorFlowProcessor.ATTRIBUTE_OUTPUT_NAME) );
			} catch (UnsupportedEncodingException e) {
				e.printStackTrace();
			}
		}
	}
}
