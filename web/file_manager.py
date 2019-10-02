import boto3

FILE_STORAGE = 'food-ai'
aws_access_key_id = 'AKIARWFZYJZWEZFTRRRI'
aws_secret_access_key = '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'


class FileManager:
    def __init__(self, file_storage=FILE_STORAGE, key=aws_access_key_id, secret_key=aws_secret_access_key):
        self.file_storage_name = file_storage
        self.key = key
        self.secret_key = secret_key
        self.s3_resource = None

    def connect_file_storage(self):
        """Used to establish AWS S3 bucket connections and instances with the provided access keys
        :return: None
        """
        try:
            if self.s3_resource is None:
                self.s3_resource = boto3.resource('s3',
                                                  aws_access_key_id=self.key,
                                                  aws_secret_access_key=self.secret_key)
        except Exception as e:
            print('connect to storage error:', str(e))

    def disconnect_file_storage(self):
        """Used to disconnect from S3 buckets.
        :return: None
        """
        try:
            if self.s3_resource is not None:
                self.s3_resource = None
        except Exception as e:
            print(str(e))

    def upload_file(self, path_to_file, file_name):
        """Used to upload file to S3 bucket for file storage.
        :param path_to_file: path of the file to upload to
        :type: str
        :param file_name: the file name you want it to be stored as on S3
        :type: str
        :return: None
        """
        self.connect_file_storage()
        try:
            self.s3_resource.Bucket(self.file_storage_name).upload_file(Filename=path_to_file, Key=file_name)
        except Exception as e:
            print(str(e))
        self.disconnect_file_storage()

    def download_file(self, file_name, path_to_download_to):
        """Download file from S3
        :param file_name: file on S3 you want to download
        :type: str
        :param path_to_download_to: location where you want to download the file to
        :type: str
        :return: None
        """
        self.connect_file_storage()
        try:
            print('downloading file from s3: {}'.format(file_name), end='...')
            self.s3_resource.Bucket(self.file_storage_name).download_file(Key=file_name, Filename=path_to_download_to)
            print('done!')
        except Exception as e:
            print('download file error:', str(e))
        self.disconnect_file_storage()

    def read_file(self, file_name):
        """Read a file from S3 without downloading. Converts the bytes data into string then
        separate it into list by newline
        :param file_name: file you want to read
        :type: str
        :return: content
        :rtype: list[str]
        """
        self.connect_file_storage()
        content = []
        try:
            binary_data = self.s3_resource.Object(self.file_storage_name, file_name).get()['Body'].read()
            content = self.process_string_data(binary_data.decode('utf-8'))
        except Exception as e:
            print(str(e))
        self.disconnect_file_storage()
        return content

    def get_list_of_files_from_storage(self):
        self.connect_file_storage()
        try:
            objects = self.s3_resource.Bucket(self.file_storage_name).objects.all()
        except Exception as e:
            print(str(e))
            return []
        self.disconnect_file_storage()
        return [obj.key for obj in objects]

    def process_string_data(self, string_data, separator='\n'):
        """Convert string data into lists using a separator. By default it is new line ('\n').
        This is used mainly to process the data we read from S3
        :param string_data: string data to process
        :type: str
        :param separator: separator to use for processing
        :type: str
        :return: list of strings separated by the specified separator
        :rtype: list[str]
        """
        return [x for x in string_data.split(separator)]


