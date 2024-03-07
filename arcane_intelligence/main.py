from configuration import GlobalConfig

if __name__ == '__main__':
  import sys
  if len(sys.argv) < 2:
      print("Usage: python config.py <config_file_path>")
      sys.exit(1)

  config_file_path = sys.argv[1]
  GlobalConfig.init_config(config_file_path)

  print(GlobalConfig.get_config('app.name'))
  print(GlobalConfig.get_config('database.host'))
  print(GlobalConfig.get_config('logging.level'))