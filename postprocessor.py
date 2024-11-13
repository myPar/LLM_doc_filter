import argparse
import os
import sys
import re
import magic


def postprocess_file(file_path: str, inplace: bool, output: str):
    def postprocess(data: str) -> str:
        data = re.sub(r"#+[ \t]*([^\n\r]+)[\r\n]+", r"\1\n", data)   # remove markdown header artifact and redundant new lines
        data = re.sub(r"\*+(\w+)\*+", r"\1", data)
        data = re.sub(r"_+(\w+)_+", r"\1", data)
        data = re.sub(r"(?<=[\n\r])---", "", data)
        data = re.sub(r"\n{2,}|(\r\n){2,}", "\n"*4, data)    # split on chunks

        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        f_data = f.read()
    if inplace:
        with open(file_path, 'w', encoding='utf-8') as f:
            pp_data = postprocess(f_data)
            f.write(pp_data)
    else:
        name, ext = os.path.splitext(os.path.basename(file_path))
        with open(os.path.join(output, name + ext), 'w', encoding='utf-8') as f:
            pp_data = postprocess(f_data)
            f.write(pp_data)


def validate_args(args):
    if args.file_path.strip() != "" and not os.path.isfile(args.file_path):  # file is specified but doesn't exist
        raise Exception(f"file - {args.file_path} doesn't exists")
    if not os.path.isdir(args.dir_path) and args.file_path.strip() == "":
        raise Exception(f"input dir - {args.dir_path} doesn't exists and no input file is specified")
    if not os.path.isdir(args.output) and not args.inplace:
        os.mkdir(args.output)


def is_text(file_path: str):
    try:
        result = magic.from_file(file_path, mime=True).split('/')[0] == 'text'
        return result
    except Exception:
        # issue only on cyrillic file names on windows
        return file_path.split(".")[-1] in ['txt', 'md']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=False, default="",
                        help='file to postprocess, if not specified, --dir_path will be used')
    parser.add_argument('--output', type=str, required=False, default="postprocess_dir",
                        help='path to postprocessing result directory; ignored if --inplace=True')
    parser.add_argument('--dir_path', type=str, required=False, default=".",
                        help='directory to get files on postprocessing from. this argument is ignored if '
                             '--file_path is specified')
    parser.add_argument('--inplace', type=bool, required=False, default=False,
                        help='weather files will be inplace postprocessed or no')

    try:
        args = parser.parse_args()
        validate_args(args)
    except Exception as e:
        print('Parse args exception: ' + repr(e), file=sys.stderr)
        parser.print_help()
        return

    output = args.output
    file_path = args.file_path.strip()
    dir_path = args.dir_path
    inplace = args.inplace

    if file_path != "":
        name, ext = os.path.splitext(os.path.basename(file_path))
        if not is_text(file_path):
            print(f"WARNING: {file_path} - is not a text file, so can't be filtered")
            return
        postprocess_file(file_path, inplace, output)
        print(f'file {name + ext} is processed')
    else:
        files = os.listdir(dir_path)
        # select only text files:
        files = [f for f in [os.path.join(dir_path, _) for _ in files] if is_text(f)]
        if len(files) == 0:
            print(f'WARNING: no text files exists here - {dir_path}, nothing to process')
            return
        for file in files:
            name, ext = os.path.splitext(os.path.basename(file_path))
            postprocess_file(file, inplace, output)
            print(f'file {name + ext} is processed')


if __name__ == '__main__':
    main()
