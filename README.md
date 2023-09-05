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

4. Run Darwin for each test trace. The HOC object hit rate output for the algorithm is printed in the standard output.
```
python3 $WORK_DIR/Darwin/algs/hierarchy-online.py -t $WORK_DIR/data/test-traces/$TRACE_NAME -m $WORK_DIR/data/models -h 100000 -d 10000000
```

5. Run other baselines for each test trace. The HOC object hit rate output for the algorithm is printed in the standard output.

```
# static baselines:
python3 $WORK_DIR/Darwin/algs/hierarchy-static-results.py -t $WORK_DIR/data/test-traces/$TRACE_NAME -o $OUTPUT_DIR -f $FREQ_THRES -s $SIZE_THRES -h 100000 -d 10000000

# percentile:
python3 $WORK_DIR/Darwin/algs/percentile.py -t $WORK_DIR/data/test-traces/$TRACE_NAME -o $OUTPUT_DIR -f 60 -s 90 -h 100000 -d 10000000 -l 100000

# hillclimbing:
python3 $WORK_DIR/Darwin/algs/hillclimbing-continuous.py -t $WORK_DIR/data/test-traces/$TRACE_NAME -o $OUTPUT_DIR -h 100000 -d 10000000 -l 500000 -c $SIZE_THRES_CLIMB_SIZE

# directmapping:
python3 $WORK_DIR/Darwin/algs/directmapping.py -t $WORK_DIR/data/test-traces/$TRACE_NAME -m $WORK_DIR/data/models -h 100000 -d 10000000

# adaptsize:
python3 $WORK_DIR/Darwin/algs/adaptsize.py -t $WORK_DIR/data/test-traces/$TRACE_NAME -o $OUTPUT_DIR -h 100000 -d 10000000 -l 100000
```

## Advanced: Train Your Own Models

This repository also include offline training code to help you generate your own cross-policy predictors and feature clustering models for HOC object hit rate. The default static experts considered are f = 2, 3, ..., 7; s = 10, 20, 50, 100, 500, 1000.

1. Generate offline hit rate data with `algs/hierarchy-static-results.py`. The raw hit and size data are stored in `$OFFLINE_OUTPUT_DIR`.

```
python3 algs/hierarchy-static-results.py -t $TRACE_FILE -o $OFFLINE_OUTPUT_DIR -f $FREQ_THRES -s $SIZE_THRES -h $HOC_SIZE -d 10000000 $DC_SIZE
```

2. Extract offline trace features with `algs/utils/traffic_model/extract_feature.py` under branch `feature-collection`.

```
python3 ./algs/utils/traffic_model/extract_feature.py $TRACE_FILE $FEATURES_OUTPUT_DIR $IAT_WINDOW $SD_WINDOW
```

3. Calculate the hit rate correlation data between experts `algs/correlation_data_gen_w_size.py`. 

```
python3 algs/correlation_data_gen_w_size.py $EXPERT0 $EXPERT1 $TRACE $FEATURES_OUTPUT_DIR_PARENT_DIR $OFFLINE_OUTPUT_DIR $CORRELATION_OUTPUT_DIR
```

4. Train the cross-policy predictors with `algs/train.py`.

```
python3 algs/train.py $HIDDEN_LAYER_SIZE $EXPERT0 $EXPERT1 $CORRELATION_OUTPUT_DIR $MODEL_OUTPUR_DIR
```

5. Train the feature clusters with `algs/cluster_ohr.py`. This requires `coarse_best_result_`$COARSE_THRESHOLD`.pkl` under $OFFLINE_RESULT_OUTPUT_DIR that contains a dictionary whose key is the trace name and the values are the best experts of this trace under the $COARSE_THRESHOLD. ({TRACE_NAME: [BEST_EXPERT]})

```
python3 algs/cluster_ohr.py $FEATURES_OUTPUT_DIR $OFFLINE_RESULT_OUTPUT_DIR $MODEL_OUTPUR_DIR 1
```