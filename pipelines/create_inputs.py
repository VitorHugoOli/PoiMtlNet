from configs.paths import IO_CHECKINS
from data.create_embb import create_embeddings
import os

from data.create_input import process_state

if __name__ == "__main__":
    # Define the path to the input file and the state name
    input_file = os.path.join(IO_CHECKINS, 'Florida.csv')
    state_name = 'florida_test'

    # Create embeddings for the specified state
    create_embeddings(state_name, input_file)
    process_state(state_name)