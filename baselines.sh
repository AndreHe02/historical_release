#!/bin/bash
# python sampler.py -e 10 -p -n no_cls -c 0 -l 1.0 -f -o
# python sampler.py -e 10 -p -n vow_con_cls -c 1 -l 1.0 -f -o
# python sampler.py -e 10 -p -n ipa_ft_cls -c 2 -l 1.0 -f -o
# python sampler.py -e 10 -p -n id_cls -c 3 -l 1.0 -f -o

# python sampler.py -e 10 -p -n ipa -c 2 -l 1.0 -f -o -m Dirichlet
# python sampler.py -e 10 -p -n ipa_rev -c 2 -l 1.0 -f -o -r -m Dirichlet
# python sampler.py -e 10 -p -n ipa_mark -c 2 -l 1.0 -f -o -m Markedness
# python sampler.py -e 10 -p -n ipa_mark_rev -c 2 -l 1.0 -f -o -r -m Markedness

python sampler.py -e 10 -p -n emb -l 1.0 -f -o -m Embedding






