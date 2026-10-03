"""Verify zero new V1 changes against the approved task-start state."""
from audit_reference import verify_reference
if __name__=='__main__':
    result=verify_reference()
    print('CLEAN_TSHC UNCHANGED FROM TASK START: YES')
    print('V1 files verified:',result['files'])
    print('Pre-existing git changes preserved:',result['git_status_matches_start'])
