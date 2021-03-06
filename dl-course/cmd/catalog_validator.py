# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import argparse
import json
import jsonschema

def parse_arguments():
    usage = 'catalog validator'
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--catalog-file', type=str, dest='catalog_file', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # スキーマそのものを検定 (不正な場合は例外が発生)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'schema', 'dataset_catalog_schema.json')
    with open(path, 'r') as fp:
        catalog_schema = json.load(fp)
    jsonschema.Draft4Validator.check_schema(catalog_schema)

    # カタログ情報のスキーマを使った検定 (不正な場合は例外が発生)
    with open(args.catalog_file, 'r') as fp:
        catalog = json.load(fp)
    jsonschema.validate(catalog, catalog_schema)

    path = os.path.split(args.catalog_file)[1]
    print('success:', path, 'is valid')
