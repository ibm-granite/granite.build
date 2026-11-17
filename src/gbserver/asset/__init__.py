#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Discover and load all the assetore classes."""

from gbserver.asset.assetstore import Assetstore

Assetstore._load_assetstore_types()
