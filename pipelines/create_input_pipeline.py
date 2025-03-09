from configs.paths import IO_CHECKINS
from data.embeddings.create_embb import create_embeddings
import os

if __name__ == "__main__":
    # Define the path to the input file and the state name
    input_file = os.path.join(IO_CHECKINS, 'checkins_Alabama.csv')
    state_name = 'Alabama'

    # Create embeddings for the specified state
    embeddings = create_embeddings(state_name, input_file)