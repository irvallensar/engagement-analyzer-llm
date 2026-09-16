import glob
import os
import shutil


def load_corpus(dir_name, ending='.tsv'):
    '''
    This is a modified version of the ldcorpus function from the corpus_toolkit
    package by Kris Kyle. https://github.com/kristopherkyle/corpus_toolkit
    '''
    # Empty list for storing corpus files
    corp = []
    # Make a list of all ".txt" files in the directory
    filenames = glob.glob(dir_name + "/*" + ending)
    print(type(filenames))
    nfiles = len(filenames)
    if nfiles == 0:
        print('''No files found. There may be a problem with your working
        directory or your file search term.''')
    fcount = 0
    # Iterate through the list of filenames
    for x in filenames:
        fcount += 1
        sm_fname = x.split(os.path.sep)[-1]
        print("Processing", sm_fname, "(" + str(fcount), "of", nfiles,
              "files)")
        # Open each file
        text = open(x, encoding='utf-8').read()
        # Remove file header and append to corpus
        corp.append(text.split('\n\n\n')[1])
    return (corp)


def format_data(corp):
    '''
    This function creates a dictionary for the corpus. Each dictionary key is
    made by joining the sentence header and the sentence text with '\n'. The
    value of a dictionary entry is a list. Each entry in the list contains
    8 dictionaries, which describe characteristics of that entry. This includes
    annotation layers.
    '''
    # Create a dict for the formatted corpus data
    data = {}
    # For each .tsv file in the corpus
    for file in corp:
        lines = file.split('\n')
        sentID = lines[0]
        items = []
        for line in lines:
            # Ignore headers and empty lines
            if len(line) < 1:
                continue
            if line[0] == "#":
                continue
            # Split lines into tab separated valuess
            tsv = line.split('\t')
            # Add filters here
            if len(tsv) < 8:
                continue
            else:
                # Append tab separated values
                items.append({"textID":tsv[0], "seqID":tsv[1],\
                "Word":tsv[2], "XPOS":tsv[3], "UPOS":tsv[4],\
                "Clause":tsv[5], "Engage":tsv[6], "Hierarchy":tsv[7],\
                "Supp":tsv[8]})
        # Assign each sent's header to its items
        data[sentID[13:]] = items
    return (data)


def AntConcDirectory(data):
    '''
    This function creates a directory for the corpus ENGAGEMENT tags to be
    loaded on AntConc. Each file in the directory is named with a modified
    sentID, and contains all the tagged words in that sent. Information about
    AntConc is available at https://www.laurenceanthony.net/software/antconc/
    '''
    # Create dict to store SentIDS
    DirDict = {}
    # Iterate through each sentence
    for sentID in data.keys():
        # Define the sent
        sent = data[sentID]
        word_tags = []
        # Iterate through each item
        for item in sent:
            # Create a label for untagged items
            if item["Engage"] == "_":
                item["Engage"] = "Ø"
            if item["Supp"] == "_":
                item["Supp"] = "Ø"
            # Define the tagged word and append it to the list
            word_tag = '_'.join([item["Word"], item["Engage"], item["Supp"]])
            word_tags.append(word_tag)
        # Separate tagged words by spaces
        DirDict[sentID] = ' '.join(word_tags)
    # Create the directory, replacing it if it already exists
    path = ('C:/Users/aaron/Working Directory/_TEST')
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)
    # Iterate through each sentID
    for sentID in DirDict.keys():
        # Have the filenames end in '.txt'
        filename = sentID + '.txt'
        # Write the file to the directory
        f = open(os.path.join(path, filename), 'w', encoding='utf-8')
        f.write(DirDict[sentID])
        f.close
    print(f"AntConc directory created at {path}")
    return


# Load the corpus (from a folder containing .tsv files)
corp = load_corpus('Engage_Completed')

# Create dictionary from corpus
data = format_data(corp)

# Create the AntConc directory
AntConcDirectory(data)
