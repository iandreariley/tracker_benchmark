from scripts import butil


def load_results(eval_type, tracker, dataset_name):
    sequence_names = butil.get_seq_names(dataset_name)
    return [butil.load_seq_result(eval_type, tracker, name) for name in sequence_names]


def load_seqs(dataset_name):
    return butil.load_seq_configs(butil.get_seq_names(dataset_name))


def main():
    # arguments
    eval_type = 'SRE'
    tracker = 'CMT'
    test_name = 'test_1_102617'
    dataset = 'tb100'
    butil.setup_seqs(dataset)
    seqs = load_seqs(dataset)

    results = load_results(eval_type, tracker, dataset)
    if len(results) > 0:
        evalResults, attrList = butil.calc_result(tracker, seqs, results, eval_type)
        print "Result of Sequences\t -- '{0}'".format(tracker)
        for seq in seqs:
            try:
                print '\t\'{0}\'{1}'.format(
                    seq.name, " "*(12 - len(seq.name))),
                print "\taveCoverage : {0:.3f}%".format(
                    sum(seq.aveCoverage)/len(seq.aveCoverage) * 100),
                print "\taveErrCenter : {0:.3f}".format(
                    sum(seq.aveErrCenter)/len(seq.aveErrCenter))
            except:
                print '\t\'{0}\'  ERROR!!'.format(seq.name)

        print "Result of attributes\t -- '{0}'".format(tracker)
        for attr in attrList:
            print "\t\'{0}\'".format(attr.name),
            print "\toverlap : {0:02.1f}%".format(attr.overlap),
            print "\tfailures : {0:.1f}".format(attr.error)

        butil.save_scores(attrList, test_name)

if __name__ == '__main__':
    main()
