from optimizer.util import jsonutil

url = 'http://omnisky:8088/ws/v1/cluster/apps?states=NEW,NEW_SAVING,SUBMITTED,ACCEPTED,RUNNING'
app_json = jsonutil.get_json(url)
print(app_json['apps'] is None)
