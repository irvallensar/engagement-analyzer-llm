import random
from bs4 import BeautifulSoup as bs
import lxml
import re
import csv
import glob
import copy
import json
import pandas as pd
import os
import shutil

example = "data/BAWE/CORPUS_UTF-8/6033g.xml"


def make_dir(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("A directory {} has been created!".format(path))


def extract_meta(soup):

    genre_families = []
    for desc in soup.sourcedesc.find_all('p'):
        if desc['n'] == "genre family":
            text = desc.text
            group = text.split("+")
            for f in group:
                genre_families.append(f.strip())
    print(genre_families)


def extract_BAWE_paragraphs(filename, id_only=True):
    holder = []
    paragrah_dict = {}
    fileid = filename.split("/")[-1]

    with open(filename, "r") as f:
        # Read each line in the file, readlines() returns a list of lines
        content = f.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        soup = bs(content, "lxml")
        print(soup)

        extract_meta(soup)

    #     for s in soup.find_all('s'):
    #         sentid = s['n']
    #         sid, pid = sentid.split(";")
    #         if "n" in pid:
    #             print(pid)
    #         else:
    #             if pid not in paragrah_dict:
    #                 paragrah_dict[pid] = []
    #             paragrah_dict[pid].append(s.text.strip())

    #         holder.append(fileid + "_" + sentid)

    # return (holder, paragrah_dict)


extract_BAWE_paragraphs(example)