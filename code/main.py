from argparse import ArgumentParser

from recompute_fitness import eval_fitness

def main():
    parser = ArgumentParser()
    parser.add_argument("-fn", "--filename", dest="filename", default="filename") 
    parser.add_argument("-a", "--arch", dest="arch", default="In_1_28_28:D_10")
    parser.add_argument("-bs", "--batch_size", dest="batch_size", default=256, type=int)
    parser.add_argument("-in", "--input", dest="input", default="MNIST")

    args = parser.parse_args()

    eval_fitness(args)
    
if __name__ == "__main__":
    main()