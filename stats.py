from utils import get_model_statistics
import sys

if __name__ == "__main__":
    try:
        model_name = sys.argv[1]
        get_model_statistics(model_name)
    except IndexError:
        print("No model provided.")
        get_model_statistics()