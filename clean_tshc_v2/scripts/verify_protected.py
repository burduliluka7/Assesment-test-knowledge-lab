"""Verify zero new protected-tree changes against the approved task-start state."""
from audit_reference import verify_reference
if __name__=='__main__':
    result=verify_reference()
    for name,row in result['trees'].items():
        print(name, 'UNCHANGED FROM TASK START:',row['unchanged'],'files:',row['files'])
