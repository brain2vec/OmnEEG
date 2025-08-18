#!/usr/bin/env python3
"""
EEG Processing Pipeline with Omneeg Integration

This script can run omneeg on either raw data or preprocessed data.
If preprocessing is requested, it will first run the preprocessing script
and then apply omneeg transformation.

Usage:
    python eeg_pipeline.py database input_folder participant output_folder [options]

Examples:
    # Run omneeg on raw data
    python eeg_pipeline.py HBN /data/raw participant_001 /output /path/to/demo.csv --mode raw
    
    # Run preprocessing first, then omneeg (uses PPSPrep/preprocessing.py)
    python eeg_pipeline.py HBN /data/raw participant_001 /output /path/to/demo.csv --mode preprocess
    
    # Run omneeg on already preprocessed data
    python eeg_pipeline.py HBN /data/preprocessed participant_001 /output /path/to/demo.csv --mode preprocessed
    
    # Use reconstruction mode (looks for labels.fif instead of RestingState_epo.fif)
    python eeg_pipeline.py HBN /data/preprocessed participant_001 /output /path/to/demo.csv --mode preprocessed --reconstruction
    
    # Use custom script locations
    python eeg_pipeline.py HBN /data/raw participant_001 /output /path/to/demo.csv --mode preprocess --preprocessing_script /path/to/custom/preprocessing.py --omneeg_script /path/to/custom/omneeg.py
"""

import sys
import os
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eeg_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EEGPipeline:
    """Main pipeline class for EEG processing with omneeg"""
    
    def __init__(self, database, input_folder, participant, output_folder, 
                 preprocessing_script='PPSPrep/preprocessing.py', omneeg_script='omneeg.py', 
                 demo_file=None, transform_module='omneeg.transform'):
        self.database = database
        self.input_folder = Path(input_folder)
        self.participant = participant
        self.output_folder = Path(output_folder)
        self.preprocessing_script = Path(preprocessing_script)
        self.omneeg_script = Path(omneeg_script)
        self.demo_file = demo_file
        self.transform_module = transform_module
        
        # Create output folder structure
        self.setup_output_folders()
    
    def setup_output_folders(self):
        """Create necessary output folder structure"""
        folders = ['raw', 'ica', 'process', 'log', 'omneeg']
        for folder in folders:
            folder_path = self.output_folder / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified folder: {folder_path}")
    
    def run_preprocessing(self, remove_channels=False):
        """Run the preprocessing script"""
        logger.info(f"Starting preprocessing for participant {self.participant}")
        
        # Check if preprocessing script exists
        if not self.preprocessing_script.exists():
            logger.error(f"Preprocessing script not found: {self.preprocessing_script}")
            logger.info("Please ensure the PPSPrep directory and preprocessing.py file exist")
            return False
        
        # Construct preprocessing command
        cmd = [
            'python', str(self.preprocessing_script),
            self.database,
            str(self.input_folder),
            self.participant,
            str(self.output_folder),
            'output_placeholder',  # This seems to be unused in the original script
            '--remove_channels', str(remove_channels).lower()
        ]
        
        logger.info(f"Running preprocessing command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Preprocessing completed successfully")
            logger.info(f"Preprocessing stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Preprocessing stderr: {result.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Preprocessing failed with return code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            logger.error(f"Standard output: {e.stdout}")
            return False
        except FileNotFoundError:
            logger.error(f"Python interpreter or preprocessing script not found")
            logger.error(f"Script path: {self.preprocessing_script}")
            return False
    
    def find_input_file(self, mode):
        """Find the appropriate input file based on processing mode"""
        if mode == 'raw':
            # Look for raw data files in input folder
            possible_extensions = ['.fif', '.set', '.edf', '.bdf', '.mat']
            for ext in possible_extensions:
                pattern = f"*{self.participant}*{ext}"
                files = list(self.input_folder.glob(pattern))
                if files:
                    return files[0]
            
            # If no specific participant file found, look for generic patterns
            for ext in possible_extensions:
                files = list(self.input_folder.glob(f"*{ext}"))
                if files:
                    logger.warning(f"No specific participant file found, using: {files[0]}")
                    return files[0]
        
        elif mode == 'preprocessed':
            # Look for preprocessed data in output folder
            ica_file = self.output_folder / 'ica' / self.participant / 'ica.fif'
            process_file = self.output_folder / 'process' / self.participant / 'RestingState_epo.fif'
            
            if process_file.exists():
                return process_file
            elif ica_file.exists():
                return ica_file
        
        elif mode == 'preprocess':
            # After preprocessing, use the processed file
            process_file = self.output_folder / 'process' / self.participant / 'RestingState_epo.fif'
            if process_file.exists():
                return process_file
            
            # Fallback to ICA file
            ica_file = self.output_folder / 'ica' / self.participant / 'ica.fif'
            if ica_file.exists():
                return ica_file
        
        return None
    
    def run_omneeg(self, input_file, output_suffix='transformed', reconstruction=False, use_transform_module=True):
        """Run omneeg transformation on the input file"""
        logger.info(f"Starting omneeg transformation on {input_file}")
        
        if use_transform_module:
            # Use the Interpolate class from omneeg/transform.py directly
            return self._run_omneeg_direct(input_file, output_suffix, reconstruction)
        else:
            # Use the original omneeg.py script
            return self._run_omneeg_script(input_file, output_suffix, reconstruction)
    
    def _run_omneeg_direct(self, input_file, output_suffix='transformed', reconstruction=False):
        """Run omneeg transformation using direct import of Interpolate class"""
        try:
            # Import required modules
            import mne
            import numpy as np
            import h5py
            from importlib import import_module
            
            # Import the Interpolate class from omneeg/transform.py
            try:
                transform_module = import_module(self.transform_module)
                Interpolate = transform_module.Interpolate
                logger.info(f"Successfully imported Interpolate from {self.transform_module}")
            except ImportError as e:
                logger.error(f"Failed to import Interpolate from {self.transform_module}: {e}")
                return None
            
            # Import demo function
            try:
                from demo import get_demo
            except ImportError as e:
                logger.error(f"Failed to import get_demo from demo module: {e}")
                return None
            
            # Construct output file path
            omneeg_output_folder = self.output_folder / 'omneeg'
            omneeg_output_folder.mkdir(parents=True, exist_ok=True)
            
            # Determine data type based on reconstruction flag
            if reconstruction:
                data_type = "labels.fif"
            else:
                data_type = 'RestingState_epo.fif'
            
            # Find the correct participant file based on database and mode
            participant_file = self._find_participant_file(data_type, reconstruction)
            if not participant_file or not participant_file.exists():
                logger.error(f"Could not find participant file: {participant_file}")
                return None
            
            logger.info(f"Loading epochs from: {participant_file}")
            
            # Load epochs
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This filename (.*) does not conform to MNE naming conventions")
                epochs = mne.read_epochs(str(participant_file), preload=True)
            
            # Resample to 500 Hz
            epochs.resample(500)
            
            # Get demographics
            df_demo = get_demo(self.database, str(self.input_folder), self.participant, self.demo_file)
            
            logger.info(f"Channel names: {epochs.info['ch_names'][:5]}...")  # Show first 5 channels
            
            # Apply interpolation transformation
            logger.info("Applying Interpolate transformation (32x32)")
            interpolate = Interpolate((32, 32))
            data_temp = interpolate(epochs)
            
            # Construct output path
            participant_id = self._get_participant_id()
            h5_path = omneeg_output_folder / f"{participant_id}.h5"
            
            # Save to HDF5
            logger.info(f"Saving transformed data to: {h5_path}")
            with h5py.File(h5_path, 'w') as f:
                # Save EEG array
                f.create_dataset('power_data', data=data_temp, compression="gzip")
                
                # Save demographics as a group of attributes (one key per column)
                demo_group = f.create_group('demographics')
                for key, value in df_demo.iloc[0].items():  # assumes 1-row per participant
                    demo_group.attrs[key] = value
            
            logger.info("Omneeg transformation completed successfully using direct method")
            return h5_path
            
        except Exception as e:
            logger.error(f"Direct omneeg transformation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _find_participant_file(self, data_type, reconstruction=False):
        """Find the participant file based on database type and processing mode"""
        if self.database == 'HBN':
            return self.output_folder / 'process' / self.participant / data_type
        
        elif self.database == 'BCAN':
            # For BCAN, we need to handle the demo file logic
            if self.demo_file:
                import pandas as pd
                demo = pd.read_csv(self.demo_file)
                basename = os.path.basename(os.path.normpath(self.participant))
                participant_id = basename.split('_', 1)[1] if '_' in basename else basename
                index_participant = demo[demo['Participant code'] == participant_id].index
                if len(index_participant) > 0:
                    Record_id = demo.iloc[index_participant[0]]['Record ID']
                    Participant_code = demo.iloc[index_participant[0]]['Participant code']
                    folder_name = f"{Record_id}_{Participant_code}"
                    return self.output_folder / 'process' / folder_name / data_type
            return self.output_folder / 'process' / self.participant / data_type
        
        elif self.database in ['VIP', 'RDB', 'ABCCT', 'XFRAGILE', 'NED', 'HSJ', 'TUEG']:
            participant_id = self.participant.split('.')[0] if self.database == 'VIP' else self.participant
            if self.database == 'RDB':
                participant_id = self.participant[:-7]
            else:
                participant_id = os.path.basename(self.participant)
            return self.output_folder / 'process' / participant_id / data_type
        
        else:
            # Default case
            return self.output_folder / 'process' / self.participant / data_type
    
    def _get_participant_id(self):
        """Get the correct participant ID based on database type"""
        if self.database == 'HBN':
            return self.participant
        elif self.database == 'BCAN':
            basename = os.path.basename(os.path.normpath(self.participant))
            return basename
        elif self.database == 'VIP':
            return self.participant.split('.')[0]
        elif self.database == 'RDB':
            return self.participant[:-7]
        else:
            return os.path.basename(self.participant)
    
    def _run_omneeg_script(self, input_file, output_suffix='transformed', reconstruction=False):
        """Run omneeg transformation using the original omneeg.py script"""
        # Check if omneeg script exists
        if not self.omneeg_script.exists():
            logger.error(f"Omneeg script not found: {self.omneeg_script}")
            return None
        
        # Construct output file path - omneeg saves as .h5 files
        omneeg_output_folder = self.output_folder / 'omneeg'
        omneeg_output_folder.mkdir(parents=True, exist_ok=True)
        
        # For omneeg, we need to determine the correct input folder based on the mode
        if 'process' in str(input_file):
            # Using preprocessed data from the process folder
            data_folder = self.output_folder / 'process'
        elif 'ica' in str(input_file):
            # Using ICA processed data
            data_folder = self.output_folder / 'ica'
        else:
            # Using raw data
            data_folder = self.input_folder
        
        # Construct omneeg command based on the actual omneeg.py script
        cmd = [
            'python', str(self.omneeg_script),
            self.database,                    # database
            str(data_folder),                 # input_folder (data_folder in omneeg)
            self.participant,                 # input_path (participant in omneeg)
            str(omneeg_output_folder),        # output_folder
            'output_placeholder',             # output_file (seems unused in omneeg)
            str(self.demo_file) if self.demo_file else 'demo_placeholder',  # demo_file
            '--reconstruction', str(reconstruction).lower()  # reconstruction flag
        ]
        
        logger.info(f"Running omneeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Omneeg saves as participant_id.h5 in the output folder
            output_file = omneeg_output_folder / f"{self.participant}.h5"
            
            logger.info("Omneeg transformation completed successfully")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Omneeg stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"Omneeg stderr: {result.stderr}")
            
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Omneeg transformation failed with return code {e.returncode}")
            logger.error(f"Error output: {e.stderr}")
            logger.error(f"Standard output: {e.stdout}")
            return None
        except FileNotFoundError:
            logger.error(f"Python interpreter or omneeg script not found")
            logger.error(f"Script path: {self.omneeg_script}")
            return None
    
    def run_pipeline(self, mode='preprocess', remove_channels=False, reconstruction=False, use_transform_module=True):
        """Run the complete pipeline based on specified mode"""
        logger.info(f"Starting EEG pipeline in '{mode}' mode for participant {self.participant}")
        
        if mode == 'preprocess':
            # Run preprocessing first
            if not self.run_preprocessing(remove_channels=remove_channels):
                logger.error("Preprocessing failed, aborting pipeline")
                return False
            
            # Find processed file
            input_file = self.find_input_file('preprocess')
            if not input_file:
                logger.error("Could not find preprocessed file after preprocessing")
                return False
            
            # Run omneeg on preprocessed data
            output_file = self.run_omneeg(input_file, 'preprocessed_transformed', 
                                        reconstruction=reconstruction, use_transform_module=use_transform_module)
            
        elif mode == 'raw':
            # Find raw input file
            input_file = self.find_input_file('raw')
            if not input_file:
                logger.error("Could not find raw input file")
                return False
            
            # Run omneeg on raw data
            output_file = self.run_omneeg(input_file, 'raw_transformed', 
                                        reconstruction=reconstruction, use_transform_module=use_transform_module)
            
        elif mode == 'preprocessed':
            # Find already preprocessed file
            input_file = self.find_input_file('preprocessed')
            if not input_file:
                logger.error("Could not find preprocessed file")
                return False
            
            # Run omneeg on preprocessed data
            output_file = self.run_omneeg(input_file, 'preprocessed_transformed', 
                                        reconstruction=reconstruction, use_transform_module=use_transform_module)
        
        else:
            logger.error(f"Unknown mode: {mode}")
            return False
        
        if output_file:
            logger.info(f"Pipeline completed successfully. Output: {output_file}")
            return True
        else:
            logger.error("Pipeline failed")
            return False

def main():
    parser = argparse.ArgumentParser(
        description='EEG Processing Pipeline with Omneeg Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Modes:
  raw          - Run omneeg directly on raw data
  preprocess   - Run preprocessing first, then omneeg
  preprocessed - Run omneeg on already preprocessed data

Examples:
  %(prog)s HBN /data/raw participant_001 /output --mode raw
  %(prog)s HBN /data/raw participant_001 /output --mode preprocess --remove_channels
  %(prog)s HBN /data/preprocessed participant_001 /output --mode preprocessed
        """
    )
    
    parser.add_argument('database', help='Database name (e.g., HBN, BCAN, etc.)')
    parser.add_argument('input_folder', help='Path to input data folder')
    parser.add_argument('participant', help='Participant identifier')
    parser.add_argument('output_folder', help='Path to output folder')
    parser.add_argument('demo_file', help='Path to demographic file')
    
    parser.add_argument('--mode', choices=['raw', 'preprocess', 'preprocessed'], 
                       default='preprocess',
                       help='Processing mode (default: preprocess)')
    
    parser.add_argument('--remove_channels', action='store_true',
                       help='Remove channels during preprocessing')
    
    parser.add_argument('--reconstruction', action='store_true',
                       help='Use reconstruction mode in omneeg (looks for labels.fif instead of RestingState_epo.fif)')
    
    parser.add_argument('--preprocessing_script', default='PPSPrep/preprocessing.py',
                       help='Path to preprocessing script (default: PPSPrep/preprocessing.py)')
    
    parser.add_argument('--omneeg_script', default='omneeg.py',
                       help='Path to omneeg script (default: omneeg.py)')
    
    parser.add_argument('--transform_module', default='omneeg.transform',
                       help='Python module path for Interpolate class (default: omneeg.transform)')
    
    parser.add_argument('--use_script', action='store_true',
                       help='Use omneeg.py script instead of direct import (default: use direct import)')
    
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run pipeline
    pipeline = EEGPipeline(
        database=args.database,
        input_folder=args.input_folder,
        participant=args.participant,
        output_folder=args.output_folder,
        preprocessing_script=args.preprocessing_script,
        omneeg_script=args.omneeg_script,
        demo_file=args.demo_file,
        transform_module=args.transform_module
    )
    
    success = pipeline.run_pipeline(
        mode=args.mode,
        remove_channels=args.remove_channels,
        reconstruction=args.reconstruction,
        use_transform_module=not args.use_script  # Use direct import by default, script if flag set
    )
    
    if success:
        logger.info("Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()