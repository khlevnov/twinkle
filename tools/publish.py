def publish(path):
    with open(f'../private/{path}') as input_file:
        with open(f'../homeworks/{path}', 'w+') as output_file:
            skip_next = False

            for line in input_file.readlines():
                spaces_count = len(line) - len(line.lstrip())
                if '# PRIVATE' in line:
                    skip_next = True

                if '# PUBLIC' in line:
                    skip_next = False
                    public_line = \
                        ' ' * spaces_count + \
                        line.split('# PUBLIC ')[1][:-1] + \
                        '  # Your code here\n'
                    output_file.write(public_line)
                    continue

                if not skip_next:
                    output_file.write(line)


files = [
    '01-vector/twinkle/linalg.py',
    '01-vector/twinkle/ops.py',
    '01-vector/twinkle/tensor.py',
    '01-vector/twinkle/tests/test_vector.py',
]

for file in files:
    publish(file)
