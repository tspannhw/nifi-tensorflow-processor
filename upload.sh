gzip -9  /Volumes/seagate/field.pem nifi-tensorflow-nar/target/nifi-tensorflow-nar-1.6.nar
scp -i /Volumes/seagate/field.pem nifi-tensorflow-nar/target/nifi-tensorflow-nar-1.6.nar.gz centos@princeton1.field.hortonworks.com:/opt/demo
#scp -i /Volumes/seagate/field.pem nifi-tensorflow-nar/target/nifi-tensorflow-nar-1.6.nar centos@princeton1.field.hortonworks.com:/opt/demo
