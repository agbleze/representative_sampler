import argparse
from ..engine.sampling_orchestrator import run_sampler
from representative_sampler import logger
from representative_sampler.core.utils.utils import discover_plugins
from representative_sampler.core.registry  import registry
import os
import yaml

_REQUIRED_CONFIG_SECTIONS = set(["sampler",
                                "scorer",
                                "embedder",
                                "importer",
                                "exporter"
                                ]
                                )


discover_plugins()


def parse_args():
    parser = argparse.ArgumentParser(description="Representative Sampler CLI")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()


def run_sampler_with_config():
    args = parse_args()
    config_path = args.config
    
    if not os.path.exists(config_path):
        msg = f"Configuration file not found: {config_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    else:
        logger.info(f"Running sampler with config: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded: {config}")
        except Exception as e:
            msg = f"Error reading configuration file: {e}"
            logger.error(msg)
            raise e

    config_keys = set(config.keys())
    if config_keys == _REQUIRED_CONFIG_SECTIONS:
        logger.info(f"All required config params are present.")
    else:
        missing_keys = _REQUIRED_CONFIG_SECTIONS - config_keys
        msg = f"Missing required config params: {missing_keys}. Required config params are: {_REQUIRED_CONFIG_SECTIONS}"
        logger.error(msg)
        raise KeyError(msg)
    logger.info(f"Successfully checked config params")
      
    logger.info("Start Reading config params")
    try:
        importer_name = config["importer"]["name"]
        importer_params = config["importer"]["params"]
        importer_status = config["importer"]["status"]
        
        embedder_name = config["embedder"]["name"]
        embedder_params = config["embedder"]["params"]
        embedder_status = config["embedder"]["status"]
        
        scorer_name = config["scorer"]["name"]
        scorer_params = config["scorer"]["params"]
        scorer_status = config["scorer"]["status"]
        
        sampler_name = config["sampler"]["name"]
        sampler_params = config["sampler"]["params"]
        sampler_status = config["sampler"]["status"]
        
        exporter_name = config["exporter"]["name"]
        exporter_params = config["exporter"]["params"]
        exporter_status = config["exporter"]["status"]  
        
    except KeyError as e:
        msg = f"Missing required config section or key: {e}"
        logger.error(msg)
        raise KeyError(msg)
    
    logger.info("Retrieving components from registry")
    importer = registry.get(name=importer_name, status=importer_status)
    embedder = registry.get(name=embedder_name, status=embedder_status)
    scorer = registry.get(name=scorer_name, status=scorer_status)
    sampler = registry.get(name=sampler_name, status=sampler_status)
    exporter = registry.get(name=exporter_name, status=exporter_status)
    logger.info("Components retrieved successfully")
    
    logger.info("Instantiating components with parameters")
    importer = importer(**importer_params)
    embedder = embedder(**embedder_params)
    scorer = scorer(**scorer_params)
    sampler = sampler(**sampler_params)
    exporter = exporter(**exporter_params)
    logger.info("Components instantiated successfully")
    
    logger.info("Running the sampling orchestrator")
    run_sampler(sampler=sampler,
                scorer=scorer,
                embedder=embedder,
                importer=importer,
                exporter=exporter
                )
    logger.info("Sampling process completed successfully")
    

if __name__ == "__main__":
    run_sampler_with_config()