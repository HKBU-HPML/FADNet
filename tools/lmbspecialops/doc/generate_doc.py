#
#  lmbspecialops - a collection of tensorflow ops
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import tensorflow as tf
from string import Template
import os
import sys


OP_DOC_TEMPLATE_STR = """\
## $name

```python
$signature
```

$summary

$description

*$hasgradient*

#### Args

$args

#### Returns

$returns

"""
OP_DOC_TEMPLATE=Template(OP_DOC_TEMPLATE_STR)

ARG_DOC_TEMPLATE_STR = """\
* ```$name```: $description

"""
ARG_DOC_TEMPLATE = Template(ARG_DOC_TEMPLATE_STR)


def camelCase_to_spaces(s):
    result = s[0].lower()
    for c in s[1:]:
        if c.islower() or c.isdigit():
            result += c
        # elif c.isupper():
        else:
            result += '_'+c.lower()
    return result

def attrValue_to_string_representation(value, value_type):
    str_fmt = "{0:0.2}"
    if value_type == 'string':
        return "'"+value.s.decode('ascii')+"'"
    elif value_type == 'bool':
        return str(value.b)
    elif value_type == 'int':
        return str(value.i)
    elif value_type == 'float':
        return str_fmt.format(value.f)
    elif value_type == 'list(int)':
        result = "[" +  ",".join(map(str,value.list.i)) + "]"
        return result
    elif value_type == 'list(float)':
        result = "[" +  ",".join(map(str_fmt.format,value.list.f)) + "]"
        return result
    else:
        return "{0}".format(value)


lib_path = sys.argv[1]
lmbspecialopslib = tf.load_op_library(lib_path)
oplist = lmbspecialopslib.OP_LIST


toc_markdown = """\
# lmbspecialops documentation

Op name | Summary
--------|--------
"""

ops_markdown = "\n"

all_op_names = [ camelCase_to_spaces(op.name) for op in oplist.op ]

for op in oplist.op:
    op_name = camelCase_to_spaces(op.name)
    if op_name.endswith('_grad'):
        continue # skip gradient ops

    has_gradient = True if op_name+'_grad' in all_op_names else False

    toc_markdown += '[{0}](#{0}) | {1}\n'.format(op_name, op.summary.strip())

    signature = op_name + '('

    args_md = str()
    for iarg in op.input_arg:
        signature += iarg.name + ', '

        args_md += ARG_DOC_TEMPLATE.safe_substitute(
                name=iarg.name,
                description=iarg.description.strip(),
                )


    for attr in op.attr:
        # skip type attributes
        if attr.type == 'type':
            continue

        arg_str = attr.name
        if hasattr(attr, 'default_value'):
            arg_str += '='+attrValue_to_string_representation(attr.default_value, attr.type)
        signature += arg_str + ', '

        args_md += ARG_DOC_TEMPLATE.safe_substitute(
                name=attr.name,
                description=attr.description.strip(),
                )

    signature = signature[:-2]+')'
    

    returns = str()

    for oarg in op.output_arg:
        if hasattr(oarg,'description'):
            returns += oarg.description


    op_md = OP_DOC_TEMPLATE.safe_substitute(
            name=op_name,
            signature=signature,
            summary=op.summary,
            description=op.description,
            hasgradient="This op has a corresponding gradient implementation" if has_gradient else "There is no corresponding gradient op",
            args=args_md,
            returns=returns,
            )

    ops_markdown += op_md




f = open('lmbspecialops_doc.md', 'w')
f.write(toc_markdown)
f.write(ops_markdown)
