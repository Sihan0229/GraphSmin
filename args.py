import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=62423, help='Random seed.')

    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--model', type=str, default='gcn',
                        choices=['sage', 'gcn'])
    parser.add_argument('--setting', type=str, default='new_Graph',
                        choices=['gnn', 'embed_smote', 'new_Graph'])
    parser.add_argument('--split_method', type=str, default='step_sampling',
                        choices=['LT_sampling', 'step_sampling'])

    parser.add_argument('--up_scale', type=float, default=1.0, help='float type from 0 to 1')
    parser.add_argument('--de_weight', type=float, default=0.001)
    parser.add_argument('--topK', type=int, default=4, help='the nearst K neighbors when construct KNN graph')
    parser.add_argument('--K', type=int, default=3, help='num of minority from 1 to 6')
    parser.add_argument('--IR', type=int, default=5, help='Imb ratio')
    parser.add_argument('--B', type=int, default=0.7, help='merge ratio')
    parser.add_argument('--LT_major', type=int, default=100, help='num of majority for DGA-2000')
    parser.add_argument('--ST_major', type=int, default=80, help='num of majority for ST DGA-2000')

    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--decoder_dim', type=int, default=128)

    parser.add_argument('-maskfeat_rate_1', type=float, default=0.1)
    parser.add_argument('-maskfeat_rate_2', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_1', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_2', type=float, default=0.1)
    return parser
