from google.cloud import storage

from pyspark import SparkContext
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()

    print('Blob {} deleted.'.format(blob_name))

def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs()

    for blob in blobs:
        print(blob.name)
def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        /a/1.txt
        /a/b/2.txt

    If you just specify prefix = '/a', you'll get back:

        /a/1.txt
        /a/b/2.txt

    However, if you specify prefix='/a' and delimiter='/', you'll get back:

        /a/1.txt

    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)
    BLBs = blobs
 #   print('Blobs:')
 #   for blob in blobs:
 #       print(blob.name)
 #   if delimiter:
 ##       print('Prefixes:')
 #       for prefix in blobs.prefixes:
 #           print(prefix)
    return BLBs
def rm_directory(DIR_name, bucket_name):
    """The function deletes the directory (DIR_name) from a given bucket (bucket_name). It is written by me, so might be buggy!"""
    blobs = list_blobs_with_prefix(bucket_name, DIR_name)
    for blob in blobs:
        delete_blob(bucket_name, blob.name)
def safeWrite_GCP(RDD, DIR_name, bucket_name):
    """Write RDD to DIR_name on bucket_name bucket, It removes this directory first."""
    rm_directory(DIR_name, bucket_name)
    RDD.saveAsTextFile("gs://%s/%s" %(bucket_name, DIR_name))
    
if __name__=="__main__":
    bucket_name = "armin-bucket"
    sc = SparkContext()
    RDD = sc.parallelize(range(10))
    safeWrite(RDD, "test-rdd", "armin-bucket")
    
    
