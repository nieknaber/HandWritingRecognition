
import sys, os
from pipeline_controller import Pipeline_Controller

def pipeline():

    pipeline = Pipeline_Controller(
        segment_dim = 16,
        window_dim = (6,3),
        num_directions = 8,

        data_directories = [
         './src/resources/original/characters/Archaic',
         './src/resources/original/characters/Hasmonean',
         './src/resources/original/characters/Herodian'
        ]
    )

    pipeline.clear_cache_results()

    # pipeline.network_training(
    #     epochs = 200
    # )

    pipeline.line_segmentation(
        images = find_files_from_arguments()
    )

    pipeline.character_classfication()

    pipeline.style_classification()
    
    
def find_files_from_arguments():
    directories = sys.argv[1]
    
    all_files = []
    files = os.listdir(directories)
    for f in files:
        if not f.startswith('.'):
            all_files.append(directories + f)
    print(all_files)
    return all_files

if __name__ == "__main__":
    pipeline()