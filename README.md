# Darwin Simulator

Darwin is a practical and provably effective CDN cache management system to learn the Hot Object Cache (HOC) admission policy that adapts to the observed traffic patterns. It employs a three-stage pipeline involving traffic pattern feature collection, unsupervised clustering for classification, and neural bandit expert
selection to choose the optimal policy.

Darwin is robust to traffic pattern changes and its design allows it to be highly customizable.
CDN operators can use the same framework for different traffic
features that can be collected, different objectives that combine
cache performance with costs, and different knob choices that are
viable in a given deployment.

More detailed information about Darwin can be found in the paper.

Darwin simulator is built on top of Cacheus simulator and Tragen traffic_modeler.

## Quick Start

1. Download the models and the test traces.

```
mkdir -p $WORK_DIR/data
cd $WORK_DIR/data
wget https://darwin-test-data.s3.us-east-2.amazonaws.com/models.zip
wget https://darwin-test-data.s3.us-east-2.amazonaws.com/test-traces.zip
```
2. Unzip.

```
unzip *.zip
```

3. Clone the repository.

```
cd $WORK_DIR
git@github.com:Janecjy/Darwin.git
```

4. Run Darwin.
```
python3 $WORK_DIR/Darwin/algs/hierarchy-online.py -t $TRACE -m $WORK_DIR/models -h 100000 -d 10000000
```