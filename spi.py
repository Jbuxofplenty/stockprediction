#!/usr/bin/env python3

import getopt, sys

def usage():
    print("""spi.py [-h | --help] [-v] -- interface for stock prediction
    
    where:
        -h | --help     show this message
        -v  version
    """)

def getopts(args):
    try:
        opts, args = getopt.getopt(args[1:], "hv:", ["help", "output="])
    except:
        print("Error: Failure to parse command line")
        usage()
        sys.exit(2)
    output = None
    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
            verbosity = a
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

def main():
    getopts(sys.argv[1:])

if __name__ == "__main__":
    main()
