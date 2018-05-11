import urllib.request
import zipfile
import click

 

@click.command()
@click.argument('download')
def down(download):
	click.echo('Downloading the dataset...')
	response = urllib.request.urlretrieve("http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip", "dataset.zip")
	
	with zipfile.ZipFile('dataset.zip',"r") as z:
		z.extractall("E:\German-Traffic-Signs-Detector\images")



#


if __name__ == '__main__':
	down()
