import alignment as a

def run():  
  vis, yfp, dsred, save_path = a.align_images()
  a.save_stacks(vis, yfp, dsred, save_path)
  
if __name__ == "__main__":
  run()
