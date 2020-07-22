from storages.backends.s3boto3 import S3Boto3Storage


class StaticStorage(S3Boto3Storage):
    location = 'static'
    default_acl = 'public-read'


class PublicMediaStorage(S3Boto3Storage):
    location = 'media'
    default_acl = 'public-read'


class ModelsStorage(S3Boto3Storage):
    location = 'models'
    default_acl = 'public-read'


class PlotStorage(S3Boto3Storage):
    location = 'media/plots'
    default_acl = 'public-read'
