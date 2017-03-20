# load h2o library
library(h2o)

### set global constants ######################################################
ip = 'localhost'         # host to connect connect to h2o server
port = 54321             # port to connect connect to h2o server
nthreads = - 1           # number of threads to use, -1 indicates max. threads 
max_mem_size = '12G'     # defines amount of available memory, per node    
dat = NULL               # input data location
col.names = NULL         # vector contain the column names, ['name1', 'name2', ...]
col.types = NULL         # vector containing the column types, ['numeric', 'enum', ...]
X = NULL                 # input variable names
y = NULL                 # target variable name
weights_column = NULL    # column name for weighting variable 
seed = 12345             # random seed, increases reproducibility
path = '/tmp'            # folder location to save java objects


### start and connect to h2o server ###########################################
h2o.init(ip = ip,
         port = port, 
         nthreads = nthreads,
         max_mem_size = max_mem_size)

### simulate data if none is available ########################################
if (is.null(dat)) {
  dat <- h2o.createFrame(rows = 200000, 
                         cols = 25, 
                         categorical_fraction = 0.2,
                         has_response = T,
                         response_factors = 2)
  y = 'response'
  X = setdiff(names(dat), y)
  
} else {
  
  dat <- h2o.importFile(dat, 
                        parse = TRUE, 
                        sep = '|', 
                        col.names = col.names, 
                        col.types = col.types)
  
}

### train straightforward GBM #################################################

# set target as factor
dat[[y]] <- as.factor(dat[[y]]) 

### partition data
splits <- h2o.splitFrame(
  data = dat, 
  ratios = c(0.6,0.2), # only need to specify 2 fractions, the 3rd is implied
  destination_frames = c("train.hex", "valid.hex", "test.hex"), 
  seed = seed
)
train <- splits[[1]]
valid <- splits[[2]]
test  <- splits[[3]]

### train gbm
# real-time info at host:port
# ~ 4 minutes for 4000 trees on mac book pro
model_id <- paste('gbm', gsub(':', '-', gsub(' ', '-', Sys.time())), sep = '-')
gbm1 = h2o.gbm(x = X, 
               y = y, 
               training_frame = train,
               validation_frame = valid,
               ntrees = 4000,
               model_id = model_id,
               max_depth = 5,
               learn_rate = 0.005, 
               min_rows = nrow(dat)*0.0025,
               weights_column = weights_column,
               seed = seed)

### report AUC
h2o.auc(h2o.performance(gbm1, newdata = train))
h2o.auc(h2o.performance(gbm1, newdata = valid))
h2o.auc(h2o.performance(gbm1, newdata = test))

### save model binary ########################################################
# h2o.saveModel(gbm1, path = path)

### save POJO (plain old ava object) #########################################
h2o.download_pojo(model = gbm1, path = path, get_jar = T)

### save MOJO (model-optimized java object) ##################################
h2o.download_mojo(model = gbm1, path = path, get_genmodel_jar = T)

### shutdown h2o server ######################################################
h2o.shutdown()
