using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using Microsoft.VisualStudio.CodeCoverage;
using Microsoft.VisualStudio;
using Microsoft.VisualStudio.Coverage.Analysis;
using System.IO;
using System.Xml.Linq;
using System.Linq;

namespace CodeCoverageCS
{
    class CoverProgram
    {
        public CoverProgram()
        {
            //set system path 
            string performance_tools_x86_path = @";C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools";
            string performance_tools_x64_path = @";C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools\x64";

            var name = "PATH";
            string pathvar = System.Environment.GetEnvironmentVariable(name);
            var value = pathvar + performance_tools_x64_path;
            var target = EnvironmentVariableTarget.Machine;
            System.Environment.SetEnvironmentVariable(name, value, target);
        }

        public void getCoverage()
        {
            //define exe path to test
            //string programPathWithArgs = @"D:\afl\mupdf\platform\win32\Release\mutool.exe clean -difa ./_pdfs/1.pdf";
            //string programPath = @"D:\afl\mupdf\platform\win32\Release\mutool.exe";
            string programPath = @"C:\Users\Morteza\Documents\Visual Studio 2015\Projects\CodeCoverageTest\Debug\CodeCoverageTest.exe";


            //set system path 
            var name = "PATH";
            string pathvar = System.Environment.GetEnvironmentVariable(name);
            var value = pathvar + @";C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools\";
            var target = EnvironmentVariableTarget.Machine;
            System.Environment.SetEnvironmentVariable(name, value, target);

            // TODO: Write code to call vsinstr.exe 
            //Process p = new Process();
            //StringBuilder sb = new StringBuilder("/COVERAGE ");
            // sb.Append(programPath);
            //p.StartInfo.FileName = @"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Team Tools\Performance Tools\vsinstr.exe";
            //p.StartInfo.Arguments = sb.ToString();
            //p.Start();
            //p.WaitForExit();
            // TODO: Look at the return code – 0 for success

            // A guid is used to keep track of the run
            Guid myrunguid = Guid.NewGuid();
            Monitor m = new Monitor();
            m.StartRunCoverage(myrunguid, @"./_coverage/cov01");


            // TODO: Launch tests that can
            // exercise myassembly.exe
            Process p2 = new Process();
            p2.StartInfo.FileName = programPath;
            //p2.StartInfo.Arguments = "clean -difa ./_pdfs/3.pdf";
            p2.Start();

            //string strCmdText;
            //strCmdText = "/C" + programPath + " clean -difa ./_pdfs/3.pdf";
            //strCmdText = "/C" + programPath;
            //Process.Start("CMD.exe", strCmdText);


            // Complete the run
            m.FinishRunCoverage(myrunguid);
        }


        public void convertCoverageFileToXmlFile(string fileName)
        {
            string coverage_file_path = @"D:\afl\mupdf\platform\win32\Release\coverage_ift\";
            string mupdf_x86_path = @"D:\afl\mupdf\platform\win32\Release\mupdf.exe";
            string mupdf_x64_path = @"D:\afl\mupdf\platform\win32\x64\Release\mupdf.exe";

            string coverage_xml_path = @"D:\afl\mupdf\platform\win32\Release\coverage_ift\temp1.coverage.coveragexml";

            using (CoverageInfo info = CoverageInfo.CreateFromFile(coverage_file_path + fileName, new string[] { mupdf_x64_path }, new string[] { }))
            {
                CoverageDS data = info.BuildDataSet();
                data.WriteXml(coverage_xml_path);
            }
            Console.WriteLine("Start2");
            readCodeCoverage(fileName);

        }


        public void readCodeCoverage(string fileName)
        {
            string coverage_file_path = @"D:\afl\mupdf\platform\win32\Release\coverage_ift\";
            string coverage_xml_path = @"D:\afl\mupdf\platform\win32\Release\coverage_ift\temp1.coverage.coveragexml";
            string coverage_template_path = @"D:\afl\mupdf\platform\win32\Release\coverage_ift\temp.xml";

            var xmlOrigin = XDocument.Load(coverage_xml_path);
            var xmlTemp = XDocument.Load(coverage_template_path);

            
            xmlTemp.Element("CoverageDSPriv").Element("Module").Element("ModuleName").Value = xmlOrigin.Element("CoverageDSPriv").Element("Module").Element("ModuleName").Value.ToString();
            xmlTemp.Element("CoverageDSPriv").Element("Module").Element("ImageSize").Value = xmlOrigin.Element("CoverageDSPriv").Element("Module").Element("ImageSize").Value.ToString();
            xmlTemp.Element("CoverageDSPriv").Element("Module").Element("ImageLinkTime").Value = xmlOrigin.Element("CoverageDSPriv").Element("Module").Element("ImageLinkTime").Value.ToString();
            xmlTemp.Element("CoverageDSPriv").Element("Module").Element("LinesCovered").Value = xmlOrigin.Element("CoverageDSPriv").Element("Module").Element("LinesCovered").Value.ToString();
   
            xmlTemp.Element("CoverageDSPriv").Element("Module").Element("LinesPartiallyCovered").Value = xmlOrigin.Element("CoverageDSPriv").Element("Module").Element("LinesPartiallyCovered").Value.ToString();
            xmlTemp.Element("CoverageDSPriv").Element("Module").Element("LinesNotCovered").Value = xmlOrigin.Element("CoverageDSPriv").Element("Module").Element("LinesNotCovered").Value.ToString();
            xmlTemp.Element("CoverageDSPriv").Element("Module").Element("BlocksCovered").Value = xmlOrigin.Element("CoverageDSPriv").Element("Module").Element("BlocksCovered").Value.ToString();
            xmlTemp.Element("CoverageDSPriv").Element("Module").Element("BlocksNotCovered").Value = xmlOrigin.Element("CoverageDSPriv").Element("Module").Element("BlocksNotCovered").Value.ToString();

            xmlTemp.Save(coverage_file_path + fileName + ".xml");
           
        }


        static void Main(string[] args)
        {
            var cp = new CoverProgram();
            //cp.getCoverage();
            Console.WriteLine("Start");
            /* for (int i=1; i<=1; i++)
            {
                 cp.convertCoverageFileToXmlFile(i + ".coverage");
            }
            */

            //cp.convertCoverageFileToXmlFile(@"host1_max_model7_div_1.0_mou_date_2019-03-16_13-53-57.coverage");
            cp.convertCoverageFileToXmlFile(@"1.coverage");
            Console.WriteLine("Finished");

        }
    }
}
