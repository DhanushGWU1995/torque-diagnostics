import { execSync } from 'node:child_process';
import { resolve } from 'node:path';

const infra = resolve(__dirname, '..');
const root = resolve(infra, '..');
const bucket = 'dats-dhanush-f2026';
const region = 'us-east-1';
const stackDir = resolve(infra, 'cdktf.out/stacks/dats-dhanush-f2026');

function run(command: string, cwd: string = infra): void {
  execSync(command, { cwd, stdio: 'inherit', env: process.env });
}

function bucketExists(): boolean {
  try {
    execSync(`aws s3api head-bucket --bucket ${bucket} --region ${region}`, {
      stdio: 'pipe',
      env: process.env,
    });
    return true;
  } catch {
    return false;
  }
}

function stateHasBucket(): boolean {
  try {
    const listed = execSync('terraform state list', {
      cwd: stackDir,
      encoding: 'utf8',
      env: process.env,
    });
    return listed.includes('aws_s3_bucket.lake');
  } catch {
    return false;
  }
}

run('npm run build --prefix admin-ui', root);
run('npx cdktf synth');

if (bucketExists() && !stateHasBucket()) {
  run(`terraform -chdir="${stackDir}" init -input=false`);
  run(`terraform -chdir="${stackDir}" import aws_s3_bucket.lake ${bucket}`);
}

run('npx cdktf deploy dats-dhanush-f2026 --auto-approve');
