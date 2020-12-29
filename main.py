
import parser



def run(args) :
    import engine

    # pick out the action.
    if args['action'] == 'train':
        engine.train(args)
    elif args['action'] == 'continue':
        engine.cont(args)
    else:
        engine.test(args)




if __name__ == '__main__':

    # get a command-line parser.
    p = parser.parser()

    # parse the command-line arguments.
    args = p.parse_args()

    # print the arguments
    print(args)

    # call the main function
    run(args)

