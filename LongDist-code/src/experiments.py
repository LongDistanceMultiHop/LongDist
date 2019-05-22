#!/usr/bin/env python3
#coding:utf-8
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Experiment Portal.
"""
'''

'''
import copy
import os
import torch
from src.parse_args import args
import src.data_utils as data_utils
import src.eval
from src.knowledge_graph import KnowledgeGraph
from src.emb.fact_network import ConvE
from src.rl.graph_search.pn import GraphSearchPolicy
from src.rl.graph_search.rs_pg import RewardShapingPolicyGradient


torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def process_data():
    data_dir = args.data_dir
    raw_kb_path = os.path.join(data_dir, 'raw.kb')
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(data_dir, 'dev.triples')
    test_path = os.path.join(data_dir, 'test.triples')
    data_utils.prepare_kb_envrioment(raw_kb_path, train_path, dev_path, test_path, args.test, args.add_reverse_relations)


def initialize_model_directory(args, random_seed=None):

    model_root_dir = args.model_root_dir
    dataset = os.path.basename(os.path.normpath(args.data_dir))

    if args.xavier_initialization:
        initialization_tag = '-xavier'
    elif args.uniform_entity_initialization:
        initialization_tag = '-uniform'
    else:
        initialization_tag = ''

    # Hyperparameter signature
    if args.model.startswith('fusion'):

        hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.baseline,
            args.entity_dim,
            args.relation_dim,
            args.history_num_layers,
            args.learning_rate,
            args.emb_dropout_rate,
            args.ff_dropout_rate,
            args.action_dropout_rate,
            args.bandwidth,
            args.beta)



    model_sub_dir = '{}-{}{}{}{}-{}'.format(
        dataset,
        args.model,
        reverse_edge_tag,
        entire_graph_tag,
        initialization_tag,
        hyperparam_sig
    )
    if args.model == 'set':
        model_sub_dir += '-{}'.format(args.beam_size)
        model_sub_dir += '-{}'.format(args.num_paths_per_entity)
    if args.relation_only:
        model_sub_dir += '-ro'
    elif args.relation_only_in_path:
        model_sub_dir += '-rpo'
    elif args.type_only:
        model_sub_dir += '-to'

    if args.test:
        model_sub_dir += '-test'

    if random_seed:
        model_sub_dir += '.{}'.format(random_seed)

    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print('Model directory created: {}'.format(model_dir))
    else:
        print('Model directory exists: {}'.format(model_dir))

    args.model_dir = model_dir

def construct_model(args):
    """
    Construct NN graph.
    """
    kg = KnowledgeGraph(args)
    if args.model.startswith('point.fusion'):
        pn = GraphSearchPolicy(args)
        fn_model = args.model.split('.')[2]
        fn_args = copy.deepcopy(args)
        fn_args.model = fn_model
        fn_args.relation_only = False
        if fn_model == 'conve':
            fn = ConvE(fn_args, kg.num_entities)
            fn_kg = KnowledgeGraph(fn_args)
        lf = RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn)
    return lf

def train(lf):
    train_path = data_utils.get_train_path(args)
    dev_path = os.path.join(args.data_dir, 'dev.triples')
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    train_data = data_utils.load_triples(
        train_path, entity_index_path, relation_index_path, group_examples_by_query=args.group_examples_by_query,
        add_reverse_relations=args.add_reversed_training_edges)
    seen_entities = set()
    dev_data = data_utils.load_triples(dev_path, entity_index_path, relation_index_path, seen_entities=seen_entities)
    if args.checkpoint_path is not None:
        lf.load_checkpoint(args.checkpoint_path)
    lf.run_train(train_data, dev_data)

def inference(lf):
    lf.batch_size = args.dev_batch_size
    lf.eval()
    lf.load_checkpoint(get_checkpoint_path(args))
    entity_index_path = os.path.join(args.data_dir, 'entity2id.txt')
    relation_index_path = os.path.join(args.data_dir, 'relation2id.txt')
    seen_entities = set()

    eval_metrics = {
        'dev': {},
        'test': {}
    }
    if args.eval_by_length_type:

        print('Test set performance:')
        lenlist=['triple4.txt','triple5.txt','triple6.txt','triple7.txt']
        test_path_list=[]
        for i in range(len(lenlist)):
            str1=os.path.join(args.data_dir, 'test/'+lenlist[i])
            test_path_list.append(str1)
        for j in range(len(test_path_list)):
            test_data = data_utils.load_triples(
                test_path_list[j], entity_index_path, relation_index_path, seen_entities=seen_entities, verbose=False)
            pred_scores = lf.forward(test_data, verbose=False)
            test_metrics = src.eval.hits_and_ranks(test_data, pred_scores, lf.kg.all_objects, verbose=True)
            eval_metrics['test']['hits_at_1'] = test_metrics[0]
            eval_metrics['test']['hits_at_3'] = test_metrics[1]
            eval_metrics['test']['hits_at_5'] = test_metrics[2]
            eval_metrics['test']['hits_at_10'] = test_metrics[3]
            eval_metrics['test']['mrr'] = test_metrics[4]
            print(str(j)+' done!!!')
            with open(args.log_file+'eval','a') as logf:
                for value in test_metrics:
                    logf.write(str(value)+'\n')

    return eval_metrics



def export_to_embedding_projector(lf):
    lf.load_checkpoint(get_checkpoint_path(args))
    lf.export_to_embedding_projector()









def get_checkpoint_path(args):
    if not args.checkpoint_path:
        return os.path.join(args.model_dir, 'model_best.tar')
    else:
        return args.checkpoint_path

def load_configs(config_path):
    with open(config_path) as f:
        print('loading configuration file {}'.format(config_path))
        for line in f:
            if not '=' in line:
                continue
            arg_name, arg_value = line.strip().split('=')
            if arg_value.startswith('"') and arg_value.endswith('"'):
                arg_value = arg_value[1:-1]
            if hasattr(args, arg_name):
                print('{} = {}'.format(arg_name, arg_value))
                arg_value2 = getattr(args, arg_name)
                if type(arg_value2) is str:
                    setattr(args, arg_name, arg_value)
                elif type(arg_value2) is bool:
                    if arg_value == 'True':
                        setattr(args, arg_name, True)
                    elif arg_value == 'False':
                        setattr(args, arg_name, False)
                    else:
                        raise ValueError('Unrecognized boolean value description: {}'.format(arg_value))
                elif type(arg_value2) is int:
                    setattr(args, arg_name, int(arg_value))
                elif type(arg_value2) is float:
                    setattr(args, arg_name, float(arg_value))
                else:
                    raise ValueError('Unrecognized attribute type: {}: {}'.format(arg_name, type(arg_value2)))
            else:
                raise ValueError('Unrecognized argument: {}'.format(arg_name))
    return args

def run_experiment(args):

    if args.test:
        args.data_dir += '.test'

    if args.process_data:

        # Process knowledge graph data

        process_data()
    else:

        initialize_model_directory(args)
        lf = construct_model(args)
        lf.cuda()
        if args.train:
            train(lf)
        elif args.inference:
            inference(lf)


if __name__ == '__main__':
    run_experiment(args)
