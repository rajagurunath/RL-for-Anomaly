# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 22:11:49 2018

@author: Arunodhaya
"""
import numpy as np
from nupic.encoders import ScalarEncoder
ScalarEncoder?

enc = ScalarEncoder(n=22, w=3, minval=2.5, maxval=97.5, clipInput=False, forced=True)
print "3 =", enc.encode(3)
print "4 =", enc.encode(4)
print "5 =", enc.encode(5)
print "1000 =", enc.encode(1000)


from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder

RandomDistributedScalarEncoder?

rdse = RandomDistributedScalarEncoder(n=21, w=3, resolution=5, offset=2.5)

print "3 =   ", rdse.encode(3)
print "4 =   ", rdse.encode(4)
print "5 =   ", rdse.encode(5)
print
print "100 = ", rdse.encode(100)
print "100000 =", rdse.encode(1000)


import datetime
from nupic.encoders.date import DateEncoder

DateEncoder?


de = DateEncoder(season=5)

now = datetime.datetime.strptime("2014-05-02 13:08:58", "%Y-%m-%d %H:%M:%S")
print "now =       ", de.encode(now)
nextMonth = datetime.datetime.strptime("2014-06-02 13:08:58", "%Y-%m-%d %H:%M:%S")
print "next month =", de.encode(nextMonth)
xmas = datetime.datetime.strptime("2014-12-25 13:08:58", "%Y-%m-%d %H:%M:%S")
print "xmas =      ", de.encode(xmas)


from nupic.encoders.category import CategoryEncoder

categories = ("cat", "dog", "monkey", "slow loris")
encoder = CategoryEncoder(w=3, categoryList=categories, forced=True)
cat = encoder.encode("cat")
dog = encoder.encode("dog")
monkey = encoder.encode("monkey")
loris = encoder.encode("slow loris")
print "cat =       ", cat
print "dog =       ", dog
print "monkey =    ", monkey
print "slow loris =", loris

print encoder.decode(cat)
ss=cat+monkey+dog
print encoder.decode(ss)
ss

from nupic.bindings.algorithms import SpatialPooler 

sp = SpatialPooler(inputDimensions=(15,),
                   columnDimensions=(4,),
                   potentialRadius=15,
                   numActiveColumnsPerInhArea=1,
                   globalInhibition=True,
                   synPermActiveInc=0.03,
                   potentialPct=1.0)
for column in xrange(4):
    connected = np.zeros((15,), dtype="int")
    sp.getConnectedSynapses(column, connected)
    print connected

output = np.zeros((4,), dtype="uint8")
sp.compute(np.array(cat,dtype='uint8'),True,output)
print output


for _ in xrange(20):
    sp.compute(cat, True, output)



from itertools import izip as zip, count

from nupic.algorithms.temporal_memory import TemporalMemory as TM
TM?


# Utility routine for printing the input vector
def formatRow(x):
  s = ''
  for c in range(len(x)):
    if c > 0 and c % 10 == 0:
      s += ' '
    s += str(x[c])
  s += ' '
  return s


# Step 1: create Temporal Pooler instance with appropriate parameters

tm = TM(columnDimensions = (50,),
        cellsPerColumn=2,
        initialPermanence=0.5,
        connectedPermanence=0.5,
        minThreshold=8,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        permanenceDecrement=0.0,
        activationThreshold=8,
        )


# Step 2: create input vectors to feed to the temporal memory. Each input vector
# must be numberOfCols wide. Here we create a simple sequence of 5 vectors
# representing the sequence A -> B -> C -> D -> E
x = np.zeros((5, tm.numberOfColumns()), dtype="uint32")
x[0, 0:10] = 1    # Input SDR representing "A", corresponding to columns 0-9
x[1, 10:20] = 1   # Input SDR representing "B", corresponding to columns 10-19
x[2, 20:30] = 1   # Input SDR representing "C", corresponding to columns 20-29
x[3, 30:40] = 1   # Input SDR representing "D", corresponding to columns 30-39
x[4, 40:50] = 1   # Input SDR representing "E", corresponding to columns 40-49


# Step 3: send this simple sequence to the temporal memory for learning
# We repeat the sequence 10 times
for i in range(10):

  # Send each letter in the sequence in order
  for j in range(5):
    activeColumns = set([i for i, j in zip(count(), x[j]) if j == 1])

    # The compute method performs one step of learning and/or inference. Note:
    # here we just perform learning but you can perform prediction/inference and
    # learning in the same step if you want (online learning).
    tm.compute(activeColumns, learn = True)

    # The following print statements can be ignored.
    # Useful for tracing internal states
    print("active cells " + str(tm.getActiveCells()))
    print("predictive cells " + str(tm.getPredictiveCells()))
    print("winner cells " + str(tm.getWinnerCells()))
    print("# of active segments " + str(tm.connections.numSegments()))

  # The reset command tells the TM that a sequence just ended and essentially
  # zeros out all the states. It is not strictly necessary but it's a bit
  # messier without resets, and the TM learns quicker with resets.
  tm.reset()


#######################################################################
#
# Step 3: send the same sequence of vectors and look at predictions made by
# temporal memory
for j in range(5):
  print "\n\n--------","ABCDE"[j],"-----------"
  print "Raw input vector : " + formatRow(x[j])
  activeColumns = set([i for i, j in zip(count(), x[j]) if j == 1])
  # Send each vector to the TM, with learning turned off
  tm.compute(activeColumns, learn = False)

  # The following print statements prints out the active cells, predictive
  # cells, active segments and winner cells.
  #
  # What you should notice is that the columns where active state is 1
  # represent the SDR for the current input pattern and the columns where
  # predicted state is 1 represent the SDR for the next expected pattern
  print "\nAll the active and predicted cells:"

  print("active cells " + str(tm.getActiveCells()))
  print("predictive cells " + str(tm.getPredictiveCells()))
  print("winner cells " + str(tm.getWinnerCells()))
  print("# of active segments " + str(tm.connections.numSegments()))

  activeColumnsIndeces = [tm.columnForCell(i) for i in tm.getActiveCells()]
  predictedColumnIndeces = [tm.columnForCell(i) for i in tm.getPredictiveCells()]


  # Reconstructing the active and inactive columns with 1 as active and 0 as
  # inactive representation.

  actColState = ['1' if i in activeColumnsIndeces else '0' for i in range(tm.numberOfColumns())]
  actColStr = ("".join(actColState))
  predColState = ['1' if i in predictedColumnIndeces else '0' for i in range(tm.numberOfColumns())]
  predColStr = ("".join(predColState))

  # For convenience the cells are grouped
  # 10 at a time. When there are multiple cells per column the printout
  # is arranged so the cells in a column are stacked together
  print "Active columns:    " + formatRow(actColStr)
  print "Predicted columns: " + formatRow(predColStr)

  # predictedCells[c][i] represents the state of the i'th cell in the c'th
  # column. To see if a column is predicted, we can simply take the OR
  # across all the cells in that column. In numpy we can do this by taking
  # the max along axis 1.


# Model Params!
MODEL_PARAMS = {
    # Type of model that the rest of these parameters apply to.
    'model': "HTMPrediction",

    # Version that specifies the format of the config.
    'version': 1,

    # Intermediate variables used to compute fields in modelParams and also
    # referenced from the control section.
    'aggregationInfo': {   'days': 0,
        'fields': [('consumption', 'sum')],
        'hours': 1,
        'microseconds': 0,
        'milliseconds': 0,
        'minutes': 0,
        'months': 0,
        'seconds': 0,
        'weeks': 0,
        'years': 0},

    'predictAheadTime': 10,

    # Model parameter dictionary.
    'modelParams': {
        # The type of inference that this model will perform
        'inferenceType': 'TemporalMultiStep',

        'sensorParams': {
            # Sensor diagnostic output verbosity control;
            # if > 0: sensor region will print out on screen what it's sensing
            # at each step 0: silent; >=1: some info; >=2: more info;
            # >=3: even more info (see compute() in py/regions/RecordSensor.py)
            'verbosity' : 0,

            # Include the encoders we use
            'encoders': {
                u'timestamp_timeOfDay': {
                    'fieldname': u'timestamp',
                    'name': u'timestamp_timeOfDay',
                    'timeOfDay': (21, 0.5),
                    'type': 'DateEncoder'
                },
                u'timestamp_dayOfWeek': None,
                u'timestamp_weekend': None,
                u'consumption': {
                    'clipInput': True,
                    'fieldname': u'consumption',
                    'maxval': 100.0,
                    'minval': 0.0,
                    'n': 50,
                    'name': u'c1',
                    'type': 'ScalarEncoder',
                    'w': 21
                },
            },

            # A dictionary specifying the period for automatically-generated
            # resets from a RecordSensor;
            #
            # None = disable automatically-generated resets (also disabled if
            # all of the specified values evaluate to 0).
            # Valid keys is the desired combination of the following:
            #   days, hours, minutes, seconds, milliseconds, microseconds, weeks
            #
            # Example for 1.5 days: sensorAutoReset = dict(days=1,hours=12),
            #
            # (value generated from SENSOR_AUTO_RESET)
            'sensorAutoReset' : None,
        },

        'spEnable': True,

        'spParams': {
            # SP diagnostic output verbosity control;
            # 0: silent; >=1: some info; >=2: more info;
            'spVerbosity' : 0,

            # Spatial Pooler implementation selector, see getSPClass
            # in py/regions/SPRegion.py for details
            # 'py' (default), 'cpp' (speed optimized, new)
            'spatialImp' : 'cpp',

            'globalInhibition': 1,

            # Number of cell columns in the cortical region (same number for
            # SP and TM)
            # (see also tpNCellsPerCol)
            'columnCount': 2048,

            'inputWidth': 0,

            # SP inhibition control (absolute value);
            # Maximum number of active columns in the SP region's output (when
            # there are more, the weaker ones are suppressed)
            'numActiveColumnsPerInhArea': 40,

            'seed': 1956,

            # potentialPct
            # What percent of the columns's receptive field is available
            # for potential synapses. At initialization time, we will
            # choose potentialPct * (2*potentialRadius+1)^2
            'potentialPct': 0.5,

            # The default connected threshold. Any synapse whose
            # permanence value is above the connected threshold is
            # a "connected synapse", meaning it can contribute to the
            # cell's firing. Typical value is 0.10. Cells whose activity
            # level before inhibition falls below minDutyCycleBeforeInh
            # will have their own internal synPermConnectedCell
            # threshold set below this default value.
            # (This concept applies to both SP and TM and so 'cells'
            # is correct here as opposed to 'columns')
            'synPermConnected': 0.1,

            'synPermActiveInc': 0.1,

            'synPermInactiveDec': 0.005,
        },

        # Controls whether TM is enabled or disabled;
        # TM is necessary for making temporal predictions, such as predicting
        # the next inputs.  Without TP, the model is only capable of
        # reconstructing missing sensor inputs (via SP).
        'tmEnable' : True,

        'tmParams': {
            # TM diagnostic output verbosity control;
            # 0: silent; [1..6]: increasing levels of verbosity
            # (see verbosity in nupic/trunk/py/nupic/research/TP.py and BacktrackingTMCPP.py)
            'verbosity': 0,

            # Number of cell columns in the cortical region (same number for
            # SP and TM)
            # (see also tpNCellsPerCol)
            'columnCount': 2048,

            # The number of cells (i.e., states), allocated per column.
            'cellsPerColumn': 32,

            'inputWidth': 2048,

            'seed': 1960,

            # Temporal Pooler implementation selector (see _getTPClass in
            # CLARegion.py).
            'temporalImp': 'cpp',

            # New Synapse formation count
            # NOTE: If None, use spNumActivePerInhArea
            #
            # TODO: need better explanation
            'newSynapseCount': 20,

            # Maximum number of synapses per segment
            #  > 0 for fixed-size CLA
            # -1 for non-fixed-size CLA
            #
            # TODO: for Ron: once the appropriate value is placed in TP
            # constructor, see if we should eliminate this parameter from
            # description.py.
            'maxSynapsesPerSegment': 32,

            # Maximum number of segments per cell
            #  > 0 for fixed-size CLA
            # -1 for non-fixed-size CLA
            #
            # TODO: for Ron: once the appropriate value is placed in TP
            # constructor, see if we should eliminate this parameter from
            # description.py.
            'maxSegmentsPerCell': 128,

            # Initial Permanence
            # TODO: need better explanation
            'initialPerm': 0.21,

            # Permanence Increment
            'permanenceInc': 0.1,

            # Permanence Decrement
            # If set to None, will automatically default to tpPermanenceInc
            # value.
            'permanenceDec' : 0.1,

            'globalDecay': 0.0,

            'maxAge': 0,

            # Minimum number of active synapses for a segment to be considered
            # during search for the best-matching segments.
            # None=use default
            # Replaces: tpMinThreshold
            'minThreshold': 9,

            # Segment activation threshold.
            # A segment is active if it has >= tpSegmentActivationThreshold
            # connected synapses that are active due to infActiveState
            # None=use default
            # Replaces: tpActivationThreshold
            'activationThreshold': 12,

            'outputType': 'normal',

            # "Pay Attention Mode" length. This tells the TM how many new
            # elements to append to the end of a learned sequence at a time.
            # Smaller values are better for datasets with short sequences,
            # higher values are better for datasets with long sequences.
            'pamLength': 1,
        },

        'clParams': {
            'regionName' : 'SDRClassifierRegion',

            # Classifier diagnostic output verbosity control;
            # 0: silent; [1..6]: increasing levels of verbosity
            'verbosity' : 0,

            # This controls how fast the classifier learns/forgets. Higher values
            # make it adapt faster and forget older patterns faster.
            'alpha': 0.005,

            # This is set after the call to updateConfigFromSubConfig and is
            # computed from the aggregationInfo and predictAheadTime.
            'steps': '1,5',

            'implementation': 'cpp',
        },

        'trainSPNetOnlyIfRequested': False,
    },
}


#import pandas as pd

from pkg_resources import resource_filename
datasetPath =resource_filename('nupic.datafiles',r'F:\RL\anomaly\yahoo_htm.csv')

print datasetPath

#with open(datasetPath) as inputFile:
#    print
#    for _ in xrange(8):
#        print inputFile.next().strip()


#from nupic.data.file_record_stream import FileRecordStream
#
#def getData():
#    return FileRecordStream(datasetPath)
#
#data = getData()
#for _ in xrange(5):
#    print data.next()
#


from nupic.frameworks.opf.model_factory import ModelFactory
model = ModelFactory.create(MODEL_PARAMS)
model.enableInference({'predictedField': 'consumption'})

records={'timestamp':"","consumption":""}
with open(datasetPath) as inputFile:
    d,c=inputFile.next().split(',')
    for _ in range(1000):
        d,c=inputFile.next().split(',')
        records['timestamp']=datetime.datetime.strptime(d,'%Y-%m-%d')
        records['consumption']=float(c)
        print "input: ", records["consumption"]
        result = model.run(records)
        print "prediction: ", result.inferences["multiStepBestPredictions"][1]

        
        













































