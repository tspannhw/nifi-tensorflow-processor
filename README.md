# nifi-tensorflow-processor
Example Tensorflow Processor using Java API for Apache NiFi 1.2+

Example using out of box TensorFlow Java example with NiFi

Article detailing creation, building and usage
https://community.hortonworks.com/content/kbentry/116803/building-a-custom-processor-in-apache-nifi-12-for.html

Currently simple classification models are supported. The model directory should have a file graph.pb containing a tensorflow graph, and a label.txt file containing labels by index of the output classes of the graph. The model must also have a variable called input which expects as Image tensor, and a output variable which stores a tensor of label index and probability. 



This is a clean update for TensorFlow 1.6.   This takes a flow file as an image 

JPG, PNG, GIF

Updated TensorFlowProcessor to TF 1.6.   Added more tests.   More cleanup.  Top 5 returned in clean naming.

Install to /usr/hdf/current/nifi/lib/

