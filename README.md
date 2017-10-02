# nifi-tensorflow-processor
Example Tensorflow Processor using Java API for Apache NiFi 1.2+

Example using out of box TensorFlow Java example with NiFi

Article detailing creation, building and usage
https://community.hortonworks.com/content/kbentry/116803/building-a-custom-processor-in-apache-nifi-12-for.html

Currently simple classification models are supported. The model directory should have a file graph.pb containing a tensorflow graph, and a label.txt file containing labels by index of the output classes of the graph. The model must also have a variable called input which expects as Image tensor, and a output variable which stores a tensor of label index and probability. 