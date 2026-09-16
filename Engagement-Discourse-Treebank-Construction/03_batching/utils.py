
def make_batch_list(sent_list: list, batch_size: int = 20):
	""" Returns two-level nested list, each for a batch of filenames.
	
	Parameters
	----------
	filenames : list
		The entire text filenames.
	batch_size : int, optional
		The number of documents for each batch. The default is 20

	Returns
	-------
	nested list of filenames.

	"""
	batches = []
	batch_count = 1
	current_batch = []
	for x, sent in enumerate(sent_list, start=1):
		if batch_count < batch_size:
			current_batch.append(sent)
			batch_count += 1
			print(x)
		else:
			current_batch.append(sent)
			batches.append(current_batch)
			print(x)
			batch_count = 1
			current_batch = []
			print('Reset')
	if len(current_batch) > 0:
		batches.append(current_batch) #this is the leftovers
	return (batches)

