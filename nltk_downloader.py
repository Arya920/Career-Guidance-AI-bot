import os

required_packages = ["stopwords", "punkt", "wordnet"]

def main():
    """
    A script to download the nltk data to the same folder that holds the other dependencies.
    It gets around an authentication error that sometimes occurs. It also has custom error messages
    intended to be helpful to people new to working on Python projects.
    """
    virtual_environment = os.environ.get("VIRTUAL_ENV") or os.environ.get(
        "CONDA_PREFIX"
    )

    if virtual_environment is None:
        print(
            "This tool is intended to be used in a virtual environment, but none were detected.\n\n \
              It looks for virtual environments by checking the $VIRTUAL_ENV and $CONDA_PREFIX \
              environment variables. Using a virtual environment for this project as a whole is \
              recommended but not required. Please create a new environment, activate the environment \
              and then install the dependencies listed in requirements.txt to the environment."
        )
        return

    try:
        import certifi
        import nltk
    except ImportError:
        print(
            "Please install the dependencies listed in requirements.txt to this environment.\n\
              Example: pip install -r requirements.txt or conda install --file requirements.txt\n"
        )
        raise

    download_dir = os.path.join(virtual_environment, "nltk_data")
    print("Will install to:", download_dir, sep="\n")
    if download_dir not in nltk.data.path:
        print(
            "That path wasn't found in the paths usef by nltk:",
            nltk.data.path,
            "aborting",
            sep="\n",
        )
        raise FileNotFoundError("nltk won't be able to find the data")

    res = nltk.download(required_packages, download_dir=download_dir)
    if not res:
        os.environ["SSL_CERT_FILE"] = certifi.where()
        res = nltk.download(required_packages, download_dir=download_dir)

    if res:
        print("The download was successfull!")
    else:
        print("The download failed!")


if __name__ == "__main__":
    main()
