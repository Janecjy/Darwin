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

## Quick Start: Test Object Hit Rate with Pretrained Models

1. Install the dependencies.

```
sudo apt install python3-pip
pip3 install bloom-filter2 torch scikit-learn numpy pynverse matplotlib 
```

2. Download the models and the test traces (used to generate figure 4a in the paper).

```
mkdir -p $WORK_DIR/data
cd $WORK_DIR/data
wget https://darwin-test-data.s3.us-east-2.amazonaws.com/models.zip
wget https://darwin-test-data.s3.us-east-2.amazonaws.com/test-traces.zip
unzip models.zip
unzip test-traces.zip
mv scratch2/09498/janechen/* ./
```

3. Clone the repository.

```
cd $WORK_DIR
git clone git@github.com:Janecjy/Darwin.git
```

4. Run Darwin for each test trace.
```
python3 $WORK_DIR/Darwin/algs/hierarchy-online.py -t $TRACE -m $WORK_DIR/data/models -h 100000 -d 10000000
```

5. Run other baselines for each test trace.

```
# static baselines:
python3 $WORK_DIR/Darwin/algs/hierarchy-static-results.py -t $TRACE -o $OUTPUT_DIR -f $FREQ_THRES -s $SIZE_THRES -h 100000 -d 10000000

# percentile:
python3 $WORK_DIR/Darwin/algs/percentile.py -t $TRACE -o $OUTPUT_DIR -f 60 -s 90 -h 100000 -d 10000000 -l 100000

# hillclimbing:

python3 $WORK_DIR/Darwin/algs/hillclimbing-continuous.py -t $TRACE -o $OUTPUT_DIR -h 100000 -d 10000000 -l 500000 -c $SIZE_THRES_CLIMB_SIZE

# directmapping:

python3 $WORK_DIR/Darwin/algs/directmapping.py -t $TRACE -m $WORK_DIR/data/models -h 100000 -d 10000000

# adaptsize:
python3 $WORK_DIR/Darwin/algs/adaptsize.py -t $TRACE -o $OUTPUT_DIR -h 100000 -d 10000000 -l 100000
```