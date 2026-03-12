import { useState } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Button,
  Paper,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  FormControl,
  InputLabel,
  Stack,
} from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import RefreshIcon from "@mui/icons-material/Refresh";
import DocumentScannerIcon from "@mui/icons-material/DocumentScanner";
import { ocrApi } from "../../api/client";

function OcrTestTab() {
  const [file, setFile] = useState<File | null>(null);
  const [provider, setProvider] = useState<string>("cloud");
  const [enhanced, setEnhanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [healthLoading, setHealthLoading] = useState(false);
  const [result, setResult] = useState<{
    text: string;
    confidence: number;
    page_count: number;
    table_count: number;
    provider: string;
    enhanced: boolean;
    metadata: object;
  } | null>(null);
  const [health, setHealth] = useState<{ cloud: boolean; onprem: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) {
      setFile(dropped);
      setResult(null);
      setError(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setResult(null);
      setError(null);
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await ocrApi.process(file, provider, enhanced);
      setResult(res.data as { text: string; confidence: number; page_count: number; table_count: number; provider: string; enhanced: boolean; metadata: object });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "OCR 처리 중 오류가 발생했습니다.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleHealthCheck = async () => {
    setHealthLoading(true);
    setError(null);
    try {
      const res = await ocrApi.health();
      setHealth(res.data as { cloud: boolean; onprem: boolean });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "헬스 체크 실패";
      setError(msg);
    } finally {
      setHealthLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom fontWeight={600}>
        OCR 테스트
      </Typography>
      <Typography variant="body2" color="text.secondary" mb={3}>
        문서 파일을 업로드하여 OCR 처리 결과를 확인합니다.
      </Typography>

      <Grid container spacing={3}>
        {/* Upload area */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                파일 업로드
              </Typography>
              <Box
                onDrop={handleDrop}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                sx={{
                  border: "2px dashed",
                  borderColor: dragOver ? "primary.main" : "divider",
                  borderRadius: 2,
                  p: 4,
                  textAlign: "center",
                  bgcolor: dragOver ? "action.hover" : "background.default",
                  cursor: "pointer",
                  transition: "all 0.2s",
                  mb: 2,
                }}
                onClick={() => document.getElementById("ocr-file-input")?.click()}
              >
                <DocumentScannerIcon sx={{ fontSize: 48, color: "text.secondary", mb: 1 }} />
                <Typography variant="body2" color="text.secondary">
                  파일을 드래그하거나 클릭하여 선택
                </Typography>
                <Typography variant="caption" color="text.disabled">
                  PDF, PNG, JPG, JPEG, TIFF, BMP 지원
                </Typography>
              </Box>
              <input
                id="ocr-file-input"
                type="file"
                accept=".pdf,.png,.jpg,.jpeg,.tiff,.tif,.bmp"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
              {file && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  선택된 파일: <strong>{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
                </Alert>
              )}

              <Stack spacing={2}>
                <FormControl size="small" fullWidth>
                  <InputLabel>OCR 프로바이더</InputLabel>
                  <Select
                    value={provider}
                    label="OCR 프로바이더"
                    onChange={(e) => setProvider(e.target.value)}
                  >
                    <MenuItem value="cloud">Cloud (Upstage)</MenuItem>
                    <MenuItem value="onprem">On-Premise</MenuItem>
                  </Select>
                </FormControl>

                <FormControlLabel
                  control={
                    <Switch
                      checked={enhanced}
                      onChange={(e) => setEnhanced(e.target.checked)}
                    />
                  }
                  label="Enhanced 모드 (고품질 처리)"
                />

                <Button
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <DocumentScannerIcon />}
                  onClick={handleProcess}
                  disabled={!file || loading}
                  fullWidth
                >
                  {loading ? "처리 중..." : "OCR 처리"}
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        {/* Health check */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                OCR 서비스 상태
              </Typography>
              <Button
                variant="outlined"
                size="small"
                startIcon={healthLoading ? <CircularProgress size={14} /> : <RefreshIcon />}
                onClick={handleHealthCheck}
                disabled={healthLoading}
                sx={{ mb: 2 }}
              >
                헬스 체크
              </Button>
              {health && (
                <Stack spacing={1}>
                  <Box display="flex" alignItems="center" gap={1}>
                    {health.cloud ? (
                      <CheckCircleIcon color="success" fontSize="small" />
                    ) : (
                      <ErrorIcon color="error" fontSize="small" />
                    )}
                    <Typography variant="body2">
                      Cloud (Upstage): {health.cloud ? "정상" : "비활성"}
                    </Typography>
                  </Box>
                  <Box display="flex" alignItems="center" gap={1}>
                    {health.onprem ? (
                      <CheckCircleIcon color="success" fontSize="small" />
                    ) : (
                      <ErrorIcon color="error" fontSize="small" />
                    )}
                    <Typography variant="body2">
                      On-Premise: {health.onprem ? "정상" : "비활성"}
                    </Typography>
                  </Box>
                </Stack>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Error */}
        {error && (
          <Grid size={{ xs: 12 }}>
            <Alert severity="error">{error}</Alert>
          </Grid>
        )}

        {/* Results */}
        {result && (
          <Grid size={{ xs: 12 }}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                  OCR 결과
                </Typography>
                <Grid container spacing={2} mb={2}>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="caption" color="text.secondary">신뢰도</Typography>
                    <Typography variant="h6" color="primary">
                      {(result.confidence * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="caption" color="text.secondary">페이지 수</Typography>
                    <Typography variant="h6">{result.page_count}</Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="caption" color="text.secondary">표 수</Typography>
                    <Typography variant="h6">{result.table_count}</Typography>
                  </Grid>
                  <Grid size={{ xs: 6, sm: 3 }}>
                    <Typography variant="caption" color="text.secondary">프로바이더</Typography>
                    <Typography variant="h6" sx={{ textTransform: "capitalize" }}>
                      {result.provider}
                    </Typography>
                  </Grid>
                </Grid>
                <Typography variant="caption" color="text.secondary" display="block" mb={1}>
                  추출된 텍스트
                </Typography>
                <Paper
                  variant="outlined"
                  sx={{
                    p: 2,
                    maxHeight: 400,
                    overflow: "auto",
                    fontFamily: "monospace",
                    fontSize: "0.8rem",
                    whiteSpace: "pre-wrap",
                    bgcolor: "background.default",
                  }}
                >
                  {result.text || "(텍스트 없음)"}
                </Paper>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}

export default OcrTestTab;
