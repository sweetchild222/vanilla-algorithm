showColumnFlag = {}

def print_table(table):

	key_list = sorted(list(table.keys()))
	tableKey = ''.join(key_list)

	showColumn = False
	if tableKey not in showColumnFlag:
	    showColumn = True
	    showColumnFlag[tableKey] = True

	template = ''

	for key in table:
		template += '{' + key + ':30}'

	if showColumn == True:
		print('')
		print('='*70)
		colmun = {}

		for key in table:
			colmun[key] = key

		print(template.format(**colmun))

		print('-'*70)

	firstKey = list(table.keys())[0]
	length = len(table[firstKey])

	for i in range(length):
		dict = {}
		for key in table:
			dict[key] = str(table[key][i])

		print(template.format(**dict))
