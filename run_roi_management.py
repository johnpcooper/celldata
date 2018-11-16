import roi_management as rm

def run():
	# create the cropped stack object
	cropped_stack = rm.run(1)
	# resize the cropped stack and save it in the current working
	# directory as "test_stack.tif"
	resized_stack = rm.run_resize(cropped_stack, "test_stack")

if __name__ == '__main__':
	run()

else:
	print("system error")