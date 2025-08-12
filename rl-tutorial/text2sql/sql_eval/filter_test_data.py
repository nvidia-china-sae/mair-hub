# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import argparse

def filter_data(input_file, output_file, database_path):
    """
    Filters test data based on the existence of schema.sql and sqlite database files.

    Args:
        input_file (str): Path to the input JSON file (e.g., test.json).
        output_file (str): Path to save the filtered JSON file.
        database_path (str): Path to the directory containing database folders.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return

    filtered_data = []
    for item in data:
        db_id = item.get('db_id')
        if not db_id:
            continue

        db_dir = os.path.join(database_path, db_id)
        schema_file = os.path.join(db_dir, 'schema.sql')
        sqlite_file = os.path.join(db_dir, f'{db_id}.sqlite')

        if os.path.exists(schema_file) and os.path.exists(sqlite_file):
            filtered_data.append(item)
        else:
            print(f"Filtering out db_id '{db_id}':")
            if not os.path.exists(schema_file):
                print(f"  - Missing: {schema_file}")
            if not os.path.exists(sqlite_file):
                print(f"  - Missing: {sqlite_file}")


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Found {len(data)} total samples.")
    print(f"Kept {len(filtered_data)} samples.")
    print(f"Filtered data saved to {output_file}")

def main():
    """Main function to parse arguments and run the filter."""
    parser = argparse.ArgumentParser(description="Filter spider dataset based on file existence.")
    parser.add_argument(
        '--input_file',
        type=str,
        default='spider_data/test.json',
        help='Path to the input test.json file.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='spider_data/filter_test.json',
        help='Path to save the filtered json file.'
    )
    parser.add_argument(
        '--db_path',
        type=str,
        default='spider_data/test_database',
        help='Path to the test_database directory.'
    )
    args = parser.parse_args()

    filter_data(args.input_file, args.output_file, args.db_path)

if __name__ == '__main__':
    main() 