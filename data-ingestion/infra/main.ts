import { createHash } from 'crypto';
import { readdirSync, readFileSync, statSync } from 'fs';
import { join, resolve } from 'path';
import { App, TerraformOutput, TerraformStack } from 'cdktf';
import { AwsProvider } from '@cdktf/provider-aws/lib/provider';
import { S3Bucket } from '@cdktf/provider-aws/lib/s3-bucket';
import { S3BucketCorsConfiguration } from '@cdktf/provider-aws/lib/s3-bucket-cors-configuration';
import { S3BucketObject } from '@cdktf/provider-aws/lib/s3-bucket-object';
import { S3BucketPolicy } from '@cdktf/provider-aws/lib/s3-bucket-policy';
import { S3BucketPublicAccessBlock } from '@cdktf/provider-aws/lib/s3-bucket-public-access-block';
import { S3BucketServerSideEncryptionConfigurationA } from '@cdktf/provider-aws/lib/s3-bucket-server-side-encryption-configuration';
import { S3BucketWebsiteConfiguration } from '@cdktf/provider-aws/lib/s3-bucket-website-configuration';
import { Construct } from 'constructs';

const BUCKET = 'dats-dhanush-f2026';
const REGION = 'us-east-1';
const DIST = resolve(__dirname, '../admin-ui/dist/admin-ui/browser');

class TorqueStack extends TerraformStack {
  constructor(scope: Construct, id: string) {
    super(scope, id);

    new AwsProvider(this, 'aws', { region: REGION });

    const lake = new S3Bucket(this, 'lake', {
      bucket: BUCKET,
      forceDestroy: false,
      tags: {
        Name: BUCKET,
        Project: 'torque-diagnostics',
      },
    });
    lake.addOverride('lifecycle.prevent_destroy', true);

    const publicAccess = new S3BucketPublicAccessBlock(this, 'public-access', {
      bucket: lake.id,
      blockPublicAcls: true,
      ignorePublicAcls: true,
      blockPublicPolicy: false,
      restrictPublicBuckets: false,
    });

    new S3BucketServerSideEncryptionConfigurationA(this, 'encrypt', {
      bucket: lake.id,
      rule: [{
        applyServerSideEncryptionByDefault: { sseAlgorithm: 'AES256' },
      }],
    });

    const site = new S3BucketWebsiteConfiguration(this, 'website', {
      bucket: lake.id,
      indexDocument: { suffix: 'index.html' },
      errorDocument: { key: 'index.html' },
    });

    new S3BucketCorsConfiguration(this, 'cors', {
      bucket: lake.id,
      corsRule: [{
        allowedHeaders: ['*'],
        allowedMethods: ['GET', 'PUT', 'POST', 'HEAD'],
        allowedOrigins: ['*'],
        exposeHeaders: ['ETag'],
        maxAgeSeconds: 3000,
      }],
    });

    const policy = new S3BucketPolicy(this, 'read-admin', {
      bucket: lake.id,
      policy: JSON.stringify({
        Version: '2012-10-17',
        Statement: [{
          Sid: 'PublicReadTorqueBay',
          Effect: 'Allow',
          Principal: '*',
          Action: 's3:GetObject',
          Resource: [
            `arn:aws:s3:::${BUCKET}/index.html`,
            `arn:aws:s3:::${BUCKET}/torque-diagnostics/admin/*`,
          ],
        }],
      }),
    });
    policy.node.addDependency(site);
    policy.node.addDependency(publicAccess);

    const redirect = `<!doctype html><meta http-equiv="refresh" content="0;url=/torque-diagnostics/admin/"><title>Torque Bay</title>`;
    new S3BucketObject(this, 'root-index', {
      bucket: lake.id,
      key: 'index.html',
      content: redirect,
      contentType: 'text/html; charset=utf-8',
      etag: createHash('md5').update(redirect).digest('hex'),
    });

    for (const file of walk(DIST)) {
      const id = `obj${createHash('sha1').update(file.key).digest('hex').slice(0, 12)}`;
      new S3BucketObject(this, id, {
        bucket: lake.id,
        key: file.key,
        source: file.abs,
        contentType: contentType(file.abs),
        etag: createHash('md5').update(readFileSync(file.abs)).digest('hex'),
      });
    }

    new TerraformOutput(this, 'website_url', {
      value: `http://${BUCKET}.s3-website-${REGION}.amazonaws.com/torque-diagnostics/admin/`,
    });
    new TerraformOutput(this, 'raw_prefix', {
      value: `s3://${BUCKET}/torque-diagnostics/raw/`,
    });
  }
}

function walk(dir: string, prefix = ''): { abs: string; key: string }[] {
  if (!statSync(dir, { throwIfNoEntry: false })?.isDirectory()) {
    throw new Error(`Angular build is missing at ${dir}. Run npm run build:ui first.`);
  }
  const found: { abs: string; key: string }[] = [];
  for (const name of readdirSync(dir)) {
    const abs = join(dir, name);
    const rel = prefix ? `${prefix}/${name}` : name;
    if (statSync(abs).isDirectory()) {
      found.push(...walk(abs, rel));
    } else {
      found.push({ abs, key: `torque-diagnostics/admin/${rel}` });
    }
  }
  return found;
}

function contentType(file: string): string {
  const ext = file.split('.').pop()?.toLowerCase();
  const types: Record<string, string> = {
    html: 'text/html; charset=utf-8',
    css: 'text/css; charset=utf-8',
    js: 'text/javascript; charset=utf-8',
    mjs: 'text/javascript; charset=utf-8',
    json: 'application/json',
    ico: 'image/x-icon',
    svg: 'image/svg+xml',
    txt: 'text/plain; charset=utf-8',
    map: 'application/json',
    woff: 'font/woff',
    woff2: 'font/woff2',
  };
  return types[ext ?? ''] ?? 'application/octet-stream';
}

const app = new App();
new TorqueStack(app, 'dats-dhanush-f2026');
app.synth();
