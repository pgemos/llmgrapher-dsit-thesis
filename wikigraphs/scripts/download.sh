#!/bin/bash
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# MODIFICATION NOTICE:
# This file was modified by Panagiotis-Konstantinos Gemos.
# Changes: - Updated dataset download URIs to use stable wikitext.smerity.com .
#          - Reorganized URLs into configuration variables.
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
#
# WikiGraphs is licensed under the terms of the Creative Commons
# Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license.
#
# WikiText-103 data (unchanged) is licensed by Salesforce.com, Inc. under the
# terms of the Creative Commons Attribution-ShareAlike 4.0 International
# (CC BY-SA 4.0) license. You can find details about CC BY-SA 4.0 at:
#
#     https://creativecommons.org/licenses/by-sa/4.0/legalcode
#
# Freebase data is licensed by Google LLC under the terms of the Creative
# Commons CC BY 4.0 license. You may obtain a copy of the License at:
#
#     https://creativecommons.org/licenses/by/4.0/legalcode
#
# ==============================================================================

# ==============================================================================
# --- USER CONFIGURATION: Panagiotis-Konstantinos Gemos ---
# ==============================================================================
# Stable wikitext-103 links
WIKITEXT_103_URL="https://wikitext.smerity.com/wikitext-103-v1.zip"
WIKITEXT_103_RAW_URL="https://wikitext.smerity.com/wikitext-103-raw-v1.zip"

# Freebase Processed Data links
FREEBASE_MAX256_URL="https://docs.google.com/uc?export=download&id=1uuSS2o72dUCJrcLff6NBiLJuTgSU-uRo"
FREEBASE_MAX512_URL="https://docs.google.com/uc?export=download&id=1nOfUq3RUoPEWNZa2QHXl2q-1gA5F6kYh"
FREEBASE_MAX1024_URL="https://docs.google.com/uc?export=download&id=1uuJwkocJXG1UcQ-RCH3JU96VsDvi7UD2"
# ==============================================================================

BASE_DIR=./data

# --- wikitext-103 ---
TARGET_DIR=${BASE_DIR}/wikitext-103
mkdir -p ${TARGET_DIR}
echo "Downloading WikiText-103..."
wget "${WIKITEXT_103_URL}" -P ${TARGET_DIR}
unzip -q ${TARGET_DIR}/wikitext-103-v1.zip -d ${TARGET_DIR}
mv ${TARGET_DIR}/wikitext-103/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/wikitext-103 ${TARGET_DIR}/wikitext-103-v1.zip

# --- wikitext-103-raw ---
TARGET_DIR=${BASE_DIR}/wikitext-103-raw
mkdir -p ${TARGET_DIR}
echo "Downloading WikiText-103-Raw..."
wget "${WIKITEXT_103_RAW_URL}" -P ${TARGET_DIR}
unzip -q ${TARGET_DIR}/wikitext-103-raw-v1.zip -d ${TARGET_DIR}
mv ${TARGET_DIR}/wikitext-103-raw/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/wikitext-103-raw ${TARGET_DIR}/wikitext-103-raw-v1.zip

# --- processed freebase graphs ---
FREEBASE_TARGET_DIR=./data
mkdir -p ${FREEBASE_TARGET_DIR}/packaged/
echo "Downloading Freebase Graphs..."

# Using --no-check-certificate to handle Google Drive redirects
wget --no-check-certificate "${FREEBASE_MAX256_URL}" -O ${FREEBASE_TARGET_DIR}/packaged/max256.tar
wget --no-check-certificate "${FREEBASE_MAX512_URL}" -O ${FREEBASE_TARGET_DIR}/packaged/max512.tar
wget --no-check-certificate "${FREEBASE_MAX1024_URL}" -O ${FREEBASE_TARGET_DIR}/packaged/max1024.tar

for version in max1024 max512 max256
do
  output_dir=${FREEBASE_TARGET_DIR}/freebase/${version}/
  mkdir -p ${output_dir}
  echo "Extracting ${version}..."
  tar -xf ${FREEBASE_TARGET_DIR}/packaged/${version}.tar -C ${output_dir}
done
rm -rf ${FREEBASE_TARGET_DIR}/packaged

echo "Success: Data processing complete."
