### load h2o library ##########################################################
library(h2o)

### set global constants ######################################################
ip <- 'localhost'            # host to connect connect to h2o server
port <- 54321                # port to connect connect to h2o server
dat <- NULL                  # input data location
col.names <- NULL            # vector contain the column names, ['name1', 'name2', ...]
col.types <- NULL            # vector containing the column types, ['numeric', 'enum', ...]
# input variable names
X <- c('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25')                  
y <- 'response'              # target variable name
# vector of categorical variables
cat_vars <- c('C4', 'C5', 'C7', 'C10', 'C14', 'C15', 'C16', 'C18', 'C21', 'C24')
tag <- 'random'              # identifier for model
seed <- 12345                # random seed, increases reproducibility
model_path <- '/tmp'         # folder location to save binary model object, on cluster 
java_path <- '/tmp'          # folder to save java artifacts, on client

### start and connect to h2o server ###########################################
h2o.init(ip = ip,
         port = port)

### import data ###############################################################
if (is.null(dat)) {
  
  # simulate data if none is available
  dat <- h2o.createFrame(rows = 200000, 
                         cols = 25, 
                         categorical_fraction = 0.4,
                         has_response = T,
                         response_factors = 2, 
                         seed = seed)
  y <- 'response'
  X <- setdiff(names(dat), y)
  
} else {
  
  # load data
  dat <- h2o.importFile(dat, 
                        parse = TRUE, 
                        sep = '|', 
                        col.names = col.names, 
                        col.types = col.types,
                        destination_frame = 'gbm_200k_grid')
  
}

### assign correct measurement roles ##########################################
for (i in 1:length(cat_vars)){
  dat[[cat_vars[i]]] <- as.factor(dat[[cat_vars[i]]])
}
dat[[y]] <- as.factor(dat[[y]]) 

### partition data ############################################################
splits <- h2o.splitFrame(
  data = dat, 
  ratios = c(0.6, 0.2), # only need to specify 2 fractions, the 3rd is implied
  destination_frames = c("train.hex", "valid.hex", "test.hex"), 
  seed = seed)
train <- splits[[1]]
valid <- splits[[2]]
test  <- splits[[3]]

### train grid search gbm #####################################################
# real-time info at host:port

### create grid_id
grid_id <- paste('gbm', gsub(':', '-', gsub(' ', '-', Sys.time())), tag, sep = '-')

### define search params
hyper_params = list(max_depth = c(2, 4, 5, 6, 10),
                    sample_rate = c(0.7, 0.8, 0.9, 1.0),
                    col_sample_rate = c(0.2, 0.4, 0.6, 0.8),
                    col_sample_rate_per_tree = seq(0.2, 1, 0.01),
                    col_sample_rate_change_per_level = seq(0.9, 1.1, 0.01),
                    min_rows = 2^seq(0, log2(nrow(train))-1, 1),
                    nbins = 2^seq(4,10,1),
                    min_split_improvement = c(0, 1e-8, 1e-6, 1e-4),
                    histogram_type = c('UniformAdaptive', 'QuantilesGlobal', 'RoundRobin'))

### define search strategy
# random more efficient 
search_criteria = list(strategy = 'RandomDiscrete',
                       max_models = 20, 
                       max_runtime_secs = 3600,         
                       seed = seed,                        
                       stopping_rounds = 5,                
                       stopping_metric = 'AUC',
                       stopping_tolerance = 1e-3)

### execute grid search
grid <- h2o.grid('gbm',
                 hyper_params = hyper_params,
                 search_criteria = search_criteria,
                 grid_id = grid_id, 
                 x = X, 
                 y = y, 
                 training_frame = train, 
                 validation_frame = valid,
                 ntrees = 10000,                                                            
                 learn_rate = 0.05, # increases convergence speed                                                     
                 learn_rate_annealing = 0.99, # increases convergence speed                                                                                                                  
                 seed = seed)
                 
### print grid of models
grid

### create list of models 
grid_models <- lapply(grid@model_ids, function(model_id) { model = h2o.getModel(model_id) })

### select best model
gbm2 <- grid_models[[1]]

### print best model
gbm2

### report AUC for best model
h2o.auc(h2o.performance(gbm2, newdata = train))
h2o.auc(h2o.performance(gbm2, newdata = valid))
h2o.auc(h2o.performance(gbm2, newdata = test))

### save model binary ########################################################
h2o.saveModel(gbm2, path = model_path)

### save POJO (plain old java object) ########################################
h2o.download_pojo(model = gbm2, path = java_path, get_jar = T)

### save MOJO (model-optimized java object) ##################################
h2o.download_mojo(model = gbm2, path = java_path, get_genmodel_jar = T)

### shutdown h2o server ######################################################
# be careful ... this will erase all of your work!
# h2o.shutdown()
