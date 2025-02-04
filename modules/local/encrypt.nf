process ENCRYPT_DIR
{
  label 'process_tiny'

  if (workflow.containerEngine == 'singularity' && !params.singularity_pull_docker_container) {
    container "/software/hgi/containers/wtsihgi-nf_ega_cryptor-1.0.sif"
  } else {
    container "mercury/wtsihgi-nf_ega_cryptor-1.0"
  }

  publishDir path: "${params.outdir}/ukbb/",
             saveAs: {filnam -> file(filnam).getName()},
             mode: 'copy',
             overwrite: 'true'

  input:
    path(input_dir)

  output:
    path("${outdir}/*", emit:encrypted_files)
    path("${outdir}", emit:encrypted_dir)
    // path("${outdir}/*/*.md5", emit:checksums)

  script:
  outdir="encryptor_output"
  """
    bash encript.sh ${input_dir} ${outdir}
  """
}
