from azureml.core import Run, Workspace, Datastore, Dataset, Experiment, ScriptRunConfig, Environment
from azureml.core.authentication import AzureCliAuthentication, InteractiveLoginAuthentication
from azure.identity import DefaultAzureCredential
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import PipelineParameter
from azureml.data.data_reference import DataReference

#This script can be used to run Caladrius on an Azure GPU

if __name__ == "__main__":
    run = Run.get_context()
    credential = DefaultAzureCredential()
    ws = Workspace.from_config()
    datastore = Datastore.get(ws, 'xview')

    # For testing:
    # input_path = OutputFileDatasetConfig(destination=(datastore, '/test_small')) # /joplin/kcenter/joplin_kcenter/
    # output_path = OutputFileDatasetConfig(destination=(datastore, '/polle/test_run/')) #/joplin/kcenter/runs/
    # distance_path = OutputFileDatasetConfig(destination=(datastore, '/polle/distance/')) #/joplin/kcenter/distance/

    # For matthew:
    input_path = OutputFileDatasetConfig(destination=(datastore, '/polle/matthew_all')) # '/trial/trial'))#'/polle/matthew_all')) #'/polle/wind/wind'))
    output_path = OutputFileDatasetConfig(destination=(datastore, 'polle/matthew_all/runs/WAAL/test/500acq/WAAL_acq500_seed555'))
    pretrained_model_path = OutputFileDatasetConfig(destination=(datastore, 'polle/wind/wind/weighted_75_epoch_pretrain/test_small-input_size_32-learning_rate_0.001-batch_size_32'))
    pretrained_model_name = 'best_model_wts.pkl'
    experiment = Experiment(workspace=ws, name='thesis_polle') #name aangepast
    config = ScriptRunConfig(source_directory='',
    script='run.py',
    compute_target='standardT4GPUlowpriority',
    arguments = [
    '--data-path', input_path,
    '--pretrained_model_path',pretrained_model_path,
    '--pretrained_model_name', pretrained_model_name,
    '--output-type', 'classification',
    #'--test',
    '--run-name', 'runname',
    '--checkpoint-path', output_path,
    # "--distance-path", distance_path,
    '--number-of-epochs', 100, #20, #50 #temporarily set to 5 as waal takes long to train
    '--batch-size',16, #usually 32, but for waal 16 due to memory usage #for test_small, use batch size 2 as there are not enough pictures for 32
    # '--weighted-loss',
    '--active','WAAL',
    '--initial_number_active',100,#100,
    '--active_images_per_iteration',500,#100,
    '--active_iterations',4, #20,
    '--MC_iterations',10,#10,
    '--num_draw_batchbald',500,
    '--tsne', True,
    '--number-of-workers',0, #decreased for WAAL, increase later given batchsize of 16
    # '--reinitialize_model', True,
    # '--num_epochs_last_actiter', 100,
    # "--learning-rate", 0.01,
    ]
    )

    env= Environment.from_conda_specification(name='caladriusenv', file_path='caladriusenv.yml') #caladriusenv2 tried (github version) as tender_beard, but did not work
    config.run_config.environment = env

    #Not: .amlignore file chooses which files not to send to azure, if there is an unnecessarily large file in there it should be added to ignore
    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(aml_url)