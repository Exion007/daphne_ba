/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <compiler/utils/CompilerUtils.h>
#include <util/ErrorHandler.h>
#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <set>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <regex>

using namespace mlir;

namespace {



    /*=========================================================*/
    // Define a set of commutative operations
    const std::set<std::string> commutativeOps = {"+", "*"};

    /**
     * @brief Represents a node in an Abstract Syntax Tree (AST).
     * 
     * This structure is used to build and compare ASTs for functions,
     * enabling detection of commutative operations.
     */
    struct ASTNode {
        std::string opName;             ///< Name of the operation
        std::vector<ASTNode> children;  ///< Child nodes representing the operands

        /**
         * @brief Constructor to initialize an ASTNode with the given operation name.
         * 
         * @param op The operation name (e.g., "+" or "*")
         */
        ASTNode(std::string op) : opName(op) {}

        /**
         * @brief Adds a child node to the current AST node.
         * 
         * @param child The child AST node to be added
         */
        void addChild(const ASTNode &child) {
            children.push_back(child);
        }

        /**
         * @brief Sorts the child nodes alphabetically if the operation is commutative.
         * 
         * This ensures that the AST representation is consistent for commutative
         * operations, allowing for proper comparison.
         */
        void sortChildrenIfNeeded() {
            if (commutativeOps.find(opName) != commutativeOps.end()) {
                // Sort the children alphabetically based on their opName
                std::sort(children.begin(), children.end(), [](const ASTNode &a, const ASTNode &b) {
                    return a.opName < b.opName;
                });
            }
        }

        /**
         * @brief Compares this ASTNode with another to check if they are similar.
         * 
         * The comparison is recursive and includes both the operation names and
         * their child nodes.
         * 
         * @param other The other ASTNode to compare with
         * @return true if both ASTs are structurally identical, false otherwise
         */
        bool isSimilar(const ASTNode &other) const {
            if (opName != other.opName || children.size() != other.children.size())
                return false;

            for (size_t i = 0; i < children.size(); ++i) {
                if (!children[i].isSimilar(other.children[i]))
                    return false;
            }
            return true;
        }
    };

    /**
     * @brief Builds an AST representation from a given function.
     * 
     * The AST captures the structure of the function's operations and their operands,
     * organizing them into a tree. This is useful for comparing function structures.
     * 
     * @param func The function to convert into an AST
     * @return An ASTNode representing the root of the function's AST
     */
    ASTNode buildASTFromFunction(func::FuncOp func) {
        ASTNode root(func.getSymName().str());

        func.walk([&](Operation *op) {
            ASTNode node(op->getName().getStringRef().str());

            // Add child nodes for operands
            for (auto operand : op->getOperands()) {
                if (auto defOp = operand.getDefiningOp()) {
                    node.addChild(ASTNode(defOp->getName().getStringRef().str()));
                }
            }

            // Sort the children if the operation is commutative
            node.sortChildrenIfNeeded();

            // Add this node to the root's children
            root.addChild(node);
        });

        return root;
    }

    /**
     * @brief Replaces constant values in a function with variables.
     * 
     * This function scans the operations of a given function and replaces any
     * constant values with newly generated variables. It adjusts the function's
     * argument list and type accordingly.
     * 
     * @param func The function to modify by replacing constants with variables
     */
    void replaceConstantsWithVariables(func::FuncOp func) {
        OpBuilder builder(func.getBody());

        // Track new argument types and body updates
        SmallVector<Type, 4> newArgTypes;
        SmallVector<Value, 4> newArgs;
        bool hasConstant = false;

        // For every argument in the function, check if it's a constant and convert it
        func.walk([&](Operation *op) {
            builder.setInsertionPoint(op);
            for (auto operand : op->getOperands()) {
                if (auto constantOp = CompilerUtils::constantOfAnyType(operand)) {
                    // Generate a new variable for the constant
                    auto loc = op->getLoc();
                    Type type = operand.getType();
                    std::string newVarName = "var" + std::to_string(newArgTypes.size());

                    // Create a new argument for the function with this variable
                    Block &body = func.getBody().front();
                    auto newArg = body.addArgument(type, loc);

                    newArgTypes.push_back(type);
                    newArgs.push_back(newArg);

                    // Replace all uses of the constant with the new variable
                    operand.replaceAllUsesWith(newArg);
                    hasConstant = true;
                }
            }
        });

        if (hasConstant) {
            // Adjust the function type with new arguments if constants were replaced
            auto funcType = func.getFunctionType();
            auto newFuncType = builder.getFunctionType(newArgTypes, funcType.getResults());
            func.setType(newFuncType);
        }
    }

    /**
     * @brief Compares two functions based on their AST structures.
     * 
     * The comparison accounts for both the operations and operands of the functions.
     * Commutative operations are normalized by sorting their operands.
     * 
     * @param func1 The first function to compare
     * @param func2 The second function to compare
     * @return true if both functions have equivalent ASTs, false otherwise
     */
    bool areFunctionsSimilarAST(func::FuncOp func1, func::FuncOp func2) {
        ASTNode ast1 = buildASTFromFunction(func1);
        ASTNode ast2 = buildASTFromFunction(func2);

        // Compare the ASTs of the two functions
        return ast1.isSimilar(ast2);
    }

    /**
     * @brief Checks for duplicate function specializations based on AST similarity.
     * 
     * This function scans a collection of functions, compares their ASTs, and removes
     * any functions that are deemed duplicates based on their structural similarity.
     * 
     * @param functions A map of function names to FuncOps representing the functions to check
     */
    void checkForDuplicateSpecializationsAST(std::unordered_map<std::string, func::FuncOp> &functions) {
        std::unordered_map<std::string, func::FuncOp> astToOriginalMap;  // To track duplicates

        for (auto it = functions.begin(); it != functions.end();) {
            const std::string &funcName = it->first;
            func::FuncOp funcOp = it->second;

            bool foundDuplicate = false;
            
            // Compare with existing functions based on AST similarity
            for (const auto &entry : astToOriginalMap) {
                func::FuncOp existingFunc = entry.second;
                if (areFunctionsSimilarAST(funcOp, existingFunc)) {
                    // A duplicate function was found
                    foundDuplicate = true;
                    break;
                }
            }

            if (foundDuplicate) {
                // Delete the duplicate function
                it = functions.erase(it);  // Erase returns the next iterator
            } else {
                // If the function is unique, store its AST
                astToOriginalMap[funcName] = funcOp;
                ++it;  // Move to the next function
            }
        }
    }

    /*=========================================================*/
    
    /**
     * @brief Checks if the function is untyped, i.e., if at least one of the inputs is
     * of unknown type.
     * 
     * @param op The `FuncOp` to check
     * @return true if `FuncOp` is untyped, false otherwise
     */
    bool isUntypedFunction(func::FuncOp op) {
        return llvm::any_of(
                op.getFunctionType().getInputs(),
                [&](Type ty) {
                    auto matTy = ty.dyn_cast<daphne::MatrixType>();
                    return
                        llvm::isa<daphne::UnknownType>(ty) ||
                        (matTy && (llvm::isa<daphne::UnknownType>(matTy.getElementType())));
                }
        );
    }

    /**
     * @brief Checks if the function is a template, by checking the types of input arguments.
     * 
     * We consider a function a template iff:
     * (1) it is an untyped function (i.e., at least one of the inputs is of unknown type
     *     or a matrix of unknown value type), or
     * (2) at least one of the inputs is a matrix with unknown properties
     * 
     * @param op The `FuncOp` to check
     * @return true if `FuncOp` is a template, false otherwise
     */
    bool isFunctionTemplate(func::FuncOp op) {
        return llvm::any_of(
                op.getFunctionType().getInputs(),
                [&](Type ty) {
                    auto matTy = ty.dyn_cast<daphne::MatrixType>();
                    return
                        llvm::isa<daphne::UnknownType>(ty) ||
                        (matTy && (
                            llvm::isa<daphne::UnknownType>(matTy.getElementType()) ||
                            (matTy.getNumRows() == -1 && matTy.getNumCols() == -1 && matTy.getSparsity() == -1)
                        ));
                }
        );
    }


    std::string uniqueSpecializedFuncName(const std::string &functionName, TypeRange inputTypes, ValueRange inputValues) {
        //static unsigned functionUniqueId = 0;
        // Creating an empty string to store the new unique specialized function name
        std::string name = functionName;

        // Iterating over types and values to use them
        for (auto it : llvm::enumerate(llvm::zip(inputTypes, inputValues))) {
            //auto index = it.index();
            auto value = std::get<1>(it.value());


            // Converting value to string
            std::string valueStr;
            llvm::raw_string_ostream valueStream(valueStr);
            value.print(valueStream);
            std::string valueName = valueStream.str();

            // Append type and value to the general name
            //std::cout << name << " | " << valueName << std::endl;
            name +=  valueName;
        }
        std::string output = functionName + '(';
        size_t pos = 0;
        bool first = true;

        // Loop through the string and find "value = "
        while ((pos = name.find("value = ", pos)) != std::string::npos) {
            if (!first) {
                output += ",";
            }
            pos += 8; // Move position past "value = "
            size_t end_pos = name.find(" ", pos);
            output += name.substr(pos, end_pos - pos);
            first = false;
        }

        output += ")";
        return output;
    }

    /**
     * @brief Extracts variable names from a given line of code.
     * 
     * This function uses a regular expression to match variable names in the input line.
     * Variable names must start with a letter or underscore and may contain letters, digits, or underscores.
     * 
     * @param line The input line of code from which to extract variable names.
     * @return A vector of strings containing the variable names found in the line.
     * 
     * @example
     * std::vector<std::string> vars = extractVariablesFromLine("int x = y + 10;");
     * // vars will contain: ["int", "x", "y"]
     */
    std::vector<std::string> extractVariablesFromLine(const std::string &line) {
        std::vector<std::string> variables;
        std::regex var_regex(R"(\b([a-zA-Z_][a-zA-Z0-9_]*)\b)");
        auto words_begin = std::sregex_iterator(line.begin(), line.end(), var_regex);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
            std::smatch match = *it;
            std::string var = match.str(1);
            variables.push_back(var);
        }
        return variables;
    }

    /**
     * @brief Normalizes commutative operations by ensuring operands are in alphabetical order.
     * 
     * This function scans a line of code for commutative operations (e.g., addition `+`)
     * and reorders the operands so that the smaller (alphabetically) operand appears first.
     * This helps to treat equivalent expressions like `a + b` and `b + a` as the same.
     * 
     * @param line The input line of code to normalize.
     * @return A string with the normalized commutative operations.
     * 
     * @example
     * std::string normalized = normalizeCommutativeOperations("y + x");
     * // normalized will be: "x + y"
     */
    std::string normalizeCommutativeOperations(const std::string &line) {
        std::regex add_regex(R"((\w+)\s*\+\s*(\w+))");
        std::smatch match;
        std::string normalizedLine = line;

        while (std::regex_search(normalizedLine, match, add_regex)) {
            std::string operand1 = match[1];
            std::string operand2 = match[2];
            if (operand1 > operand2) std::swap(operand1, operand2);
            std::string replacement = operand1 + " + " + operand2;
            normalizedLine = match.prefix().str() + replacement + match.suffix().str();
        }

        return normalizedLine;
    }

    /**
     * @brief Normalizes the entire function body by replacing variables, constants, 
     * and commutative operations, and ignoring non-essential lines.
     * 
     * This function processes each line of a function body, normalizing variable names,
     * replacing constants with "CONST", and reordering operands in commutative operations.
     * It also skips over non-essential lines, like those containing logging statements.
     * 
     * @param body A vector of strings, where each string is a line of the function body.
     * @return A vector of strings representing the normalized function body.
     * 
     * @example
     * std::vector<std::string> body = {"int x = 5 + y;", "log('test');", "float z = x + 10;"};
     * std::vector<std::string> normalizedBody = normalizeFunctionBody(body);
     * // normalizedBody will be: {"TYPE var0 = CONST + var1;", "TYPE var2 = var0 + CONST;"}
     */
    std::vector<std::string> normalizeFunctionBody(const std::vector<std::string> &body) {
        std::unordered_map<std::string, std::string> variableMapping;
        int variableCounter = 0;
        std::vector<std::string> normalizedBody;

        for (const auto &line : body) {
            std::string normalizedLine = line;

            // Ignore non-essential lines like logging
            if (line.find("log") != std::string::npos) {
                continue; // Skip lines that include "log" (for simplicity)
            }

            // Normalize data types (replace 'int' and 'float' with 'TYPE')
            normalizedLine = std::regex_replace(normalizedLine, std::regex(R"(\b(int|float|double|char|long)\b)"), "TYPE");

            // Normalize variables
            std::vector<std::string> variables = extractVariablesFromLine(line);
            for (const auto &var : variables) {
                if (variableMapping.find(var) == variableMapping.end()) {
                    variableMapping[var] = "var" + std::to_string(variableCounter++);
                }
                // Use word boundaries to replace exact matches
                normalizedLine = std::regex_replace(normalizedLine, std::regex("\\b" + var + "\\b"), variableMapping[var]);
            }

            // Normalize constants (replace numeric literals with "CONST")
            normalizedLine = std::regex_replace(normalizedLine, std::regex(R"(\b\d+\b)"), "CONST");

            // Normalize commutative operations
            normalizedLine = normalizeCommutativeOperations(normalizedLine);

            // Remove extra whitespace
            normalizedLine = std::regex_replace(normalizedLine, std::regex(R"(\s+)"), " ");

            normalizedBody.push_back(normalizedLine);
        }

        return normalizedBody;
    }

    /**
     * @brief Compares two function bodies after normalization to check if they are similar.
     * 
     * This function normalizes both function bodies (replacing variables, constants, 
     * and normalizing commutative operations) and then checks if the normalized versions
     * of both functions are identical.
     * 
     * @param body1 The first function body (vector of strings).
     * @param body2 The second function body (vector of strings).
     * @return true if the two function bodies are similar after normalization, false otherwise.
     * 
     * @example
     * std::vector<std::string> body1 = {"int a = b + c;", "float d = a + 10;"};
     * std::vector<std::string> body2 = {"float x = y + z;", "int w = x + 5;"};
     * bool areSimilar = areFunctionsSimilar(body1, body2);
     * // areSimilar will be true because the functions are similar after normalization.
     */
    bool areFunctionsSimilar(const std::vector<std::string> &body1, const std::vector<std::string> &body2) {
        std::vector<std::string> normalizedBody1 = normalizeFunctionBody(body1);
        std::vector<std::string> normalizedBody2 = normalizeFunctionBody(body2);

        return normalizedBody1.size() == normalizedBody2.size() &&
            std::equal(normalizedBody1.begin(), normalizedBody1.end(), normalizedBody2.begin());
    }

    /**
     * @brief Extracts the body of a function as a vector of strings.
     * 
     * This function walks through each operation in the function and collects 
     * the operation as a string in a vector. Each operation corresponds to a line in the function.
     * 
     * @param func The function from which to extract the body (as a `FuncOp`).
     * @return A vector of strings where each string represents a line of the function body.
     * 
     * @example
     * func::FuncOp func = ...; // Assuming func is initialized elsewhere
     * std::vector<std::string> body = getFunctionBody(func);
     * // body might look like: {"int foo(int x) {", "return x + 5;", "}"}
     */
    std::vector<std::string> getFunctionBody(func::FuncOp func) {
        std::vector<std::string> body;
        func.walk([&](Operation *op) {
            std::string line;
            llvm::raw_string_ostream os(line);
            op->print(os);
            body.push_back(line);
        });
        return body;
    }

    /**
     * @brief Checks if specialized versions of functions already exist by comparing their normalized bodies.
     * 
     * This function iterates through a map of functions, normalizes their bodies, and compares 
     * the normalized bodies. If a similar function (with the same normalized body) already exists, 
     * it skips the specialization process for that function.
     * 
     * @param functions A map where the key is the function name and the value is the function (`FuncOp`).
     * 
     * @example
     * std::unordered_map<std::string, func::FuncOp> functions = ...;
     * checkForDuplicateSpecializations(functions);
     * // This will identify and skip functions that are redundant due to having similar bodies.
     */
    void checkForDuplicateSpecializations(std::unordered_map<std::string, func::FuncOp> &functions) {
        std::unordered_map<std::string, std::string> normalizedToOriginalMap; // To track duplicates

        for (auto it = functions.begin(); it != functions.end();) {
            const std::string &funcName = it->first;
            func::FuncOp funcOp = it->second;

            std::vector<std::string> funcBody = getFunctionBody(funcOp);
            std::vector<std::string> normalizedBody = normalizeFunctionBody(funcBody);

            // Convert normalized body to a single string for easy comparison
            std::string normalizedBodyStr;
            for (const auto &line : funcBody) {
                if(line.find("daphne.generic_call") == std::string::npos){
                    normalizedBodyStr += line + "\n"; 
                }
            }

            //std::cout << "Function: " << funcName << "Body String: ";
            //std::cout << normalizedBodyStr << std::endl << std::endl;
            // Check if this normalized body already exists
            if (normalizedToOriginalMap.find(normalizedBodyStr) != normalizedToOriginalMap.end()) {
                std::string existingFuncName = normalizedToOriginalMap[normalizedBodyStr];
                //std::cout << "Function " << funcName << " is similar to " << existingFuncName << ". Deleting this function." << std::endl;
                // Erase the current function as it is a duplicate
                it = functions.erase(it); // Erase returns the next iterator
            } else {
                // If the function is unique, store its normalized body
                normalizedToOriginalMap[normalizedBodyStr] = funcName;
                ++it; // Move to the next function
            }
        }
    }
    /**
     * @brief Check if a function with the given input/output types can be called with the input types given.
     * @param functionType The type of the function
     * @param callTypes The types used in the call
     * @return true if the types match for a call, false otherwise
     */
    bool callTypesMatchFunctionTypes(FunctionType functionType, TypeRange callTypes) {
        for(auto zipIt : llvm::zip(functionType.getInputs(), callTypes)) {
            auto funcTy = std::get<0>(zipIt);
            auto callTy = std::get<1>(zipIt);
            // Note that we explicitly take all properties (e.g., shape) into account.
            if(funcTy != callTy)
                return false;
        }
        return true;
    }

    /**
     * @brief Get argument types for the specialized version of a template function.
     * @param functionType The types of the template function.
     * @param callTypes The types used in the call to the specialized version.
     * @param funcName The name of the function to call
     * @param callLoc The location of the call
     * @return The argument types to use for the specialized version
     */
    std::vector<Type> getSpecializedFuncArgTypes(FunctionType functionType, TypeRange callTypes, const std::string & funcName, mlir::Location callLoc) {
        auto unknownTy = daphne::UnknownType::get(functionType.getContext());
        std::vector<mlir::Type> specializedTypes;
        for(auto it : llvm::enumerate(llvm::zip(functionType.getInputs(), callTypes))) {
            auto index = it.index();
            auto funcInTy = std::get<0>(it.value());
            auto specializedTy = std::get<1>(it.value());
            if(funcInTy != specializedTy) {
                auto funcMatTy = funcInTy.dyn_cast<daphne::MatrixType>();
                auto specializedMatTy = specializedTy.dyn_cast<daphne::MatrixType>();
                bool isMatchingUnknownMatrix =
                    funcMatTy && specializedMatTy && funcMatTy.getElementType() == unknownTy;
                bool isMatchingUnknownPropertiesMatrix =
                    funcMatTy && specializedMatTy && funcMatTy.getElementType() == specializedMatTy.getElementType() &&
                    funcMatTy.getNumRows() == -1 && funcMatTy.getNumCols() == -1 && funcMatTy.getSparsity() == -1;
                if(!isMatchingUnknownMatrix && !isMatchingUnknownPropertiesMatrix && funcInTy != unknownTy) {
                    std::string s;
                    llvm::raw_string_ostream stream(s);
                    // TODO The function name funcName has a cryptic suffix from overloading/specialization, which is not suitable for users for see.
                    // TODO This error message can shiw up even for typed functions which are no "templates", which is confusing for a user.
                    // TODO The index seems to be off by 1 (too large)... (or not, simply 0-based counting).
                    stream << "call to function template `" << funcName << "` with invalid types for argument " << index
                           << ": expected `" << funcInTy << "`, got `" << specializedTy << "`";
                    throw ErrorHandler::compilerError(callLoc, "SpecializeGenericFunctionsPass", stream.str());
                }
            }
            // Note that specializedTy may explicitly contain property information (e.g., shape).
            specializedTypes.push_back(specializedTy);
        }
        return specializedTypes;
    }

    /**
     * @brief Set the result types to the types of the function results.
     * @param results The results for which to fix the types
     * @param functionType The function type
     * @return true if changes where made, else false
     */
    bool fixResultTypes(ResultRange results, FunctionType functionType) {
        bool madeChanges = false;
        for(auto it : llvm::zip(results, functionType.getResults())) {
            auto result = std::get<0>(it);
            auto functionResultTy = std::get<1>(it);
            if(result.getType() != functionResultTy) {
                madeChanges = true;
                result.setType(functionResultTy);
            }
        }
        return madeChanges;
    }

    /**
     * @brief Run partial type and label inference on the given `FuncOp`.
     * @param function The `FuncOp`
     * @return The inferred `FuncOp` (same as input), or `nullptr` if an error happened
     */
    func::FuncOp inferTypesInFunction(func::FuncOp function) {
        // Run inference
        mlir::PassManager pm(function->getContext(), "func.func");
        pm.enableVerifier(false);
        // TODO There is a cyclic dependency between (shape) inference and
        // constant folding (included in canonicalization), at the moment we
        // run only three iterations of both passes (see #173).
        pm.addPass(daphne::createInferencePass({true, true, true, true, true}));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(daphne::createInferencePass({true, true, true, true, true}));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(daphne::createInferencePass({true, true, true, true, true}));
        pm.addPass(createCanonicalizerPass());
        pm.addPass(daphne::createInferencePass({true, true, true, true, true}));
        pm.addPass(createCanonicalizerPass());
        if(failed(pm.run(function))) {
            throw ErrorHandler::compilerError(
                function.getOperation(), "SpecializeGenericFunctionsPass",
                "could not infer types for a call of function template: " +
                    function.getName().str());
        }
        return function;
    }

    class SpecializeGenericFunctionsPass
        : public PassWrapper<SpecializeGenericFunctionsPass, OperationPass<ModuleOp>> {
        std::unordered_map<std::string, func::FuncOp> functions;
        std::multimap<std::string, func::FuncOp> specializedVersions;
        std::set<func::FuncOp> visited;
        std::set<func::FuncOp> called;
        std::set<func::FuncOp> templateFunctions;

        std::map<std::string, std::set<std::string>> callGraph;
        std::set<std::set<std::string>> recursiveCalls;
        std::map<std::set<std::string>, int> recursiveCallsNum;
        std::set<std::string> visitedInGraph;
        std::vector<std::string> callStack;
        std::map<std::string, std::string> duplicateFunctions;

        const DaphneUserConfig& userConfig;
        std::shared_ptr<spdlog::logger> logger;

    public:
        explicit SpecializeGenericFunctionsPass(const DaphneUserConfig& cfg) : userConfig(cfg) {
            logger = spdlog::get("compiler");
        }

    private:
        void detectRecursion(const std::string &func, std::set<std::string> &visitedInGraph, std::vector<std::string> &callStack) {
            // If function is already on the stack, we found a recursion
            auto it = std::find(callStack.begin(), callStack.end(), func);
            if (it != callStack.end()) {
                // Extract the recursion cycle
                std::set<std::string> cycle(it, callStack.end());
                recursiveCalls.insert(cycle);
                return;
            }

            // If function was already visited and didn't form a cycle, skip it
            if (visitedInGraph.find(func) != visitedInGraph.end()) {
                return;
            }

            // Mark the function as visited and add it to the current call stack
            visitedInGraph.insert(func);
            callStack.push_back(func);

            // Recursively visit all called functions
            if (callGraph.find(func) != callGraph.end()) {
                for (const auto &calledFunc : callGraph[func]) {
                    detectRecursion(calledFunc, visitedInGraph, callStack);
                }
            }

            // Backtrack: remove the function from the call stack
            callStack.pop_back();
        }

        // Function to initiate recursion detection for all functions in the call graph
        void findRecursions() {
            // Clear previous recursive calls
            recursiveCalls.clear();

            // Iterate over each function in the call graph
            for (const auto &entry : callGraph) {
                // Reset visited and call stack for each new starting function
                std::set<std::string> visitedInGraph;
                std::vector<std::string> callStack;
                
                // Detect recursions starting from the current function
                detectRecursion(entry.first, visitedInGraph, callStack);
            }
        }
        /**
         * @brief Print the callgraph  -> Debugging Purposes!!     
         */
        void printCallGraph() {
            for(const auto &entry : callGraph) {
                std::string funcName = entry.first;
                //std::cout << funcName << " #!#!#calls:#!#!# ";
                if(entry.second.empty()) {
                    //std::cout << "No functions";
                } else {
                    for (const std::string &calledFuncName : entry.second) {
                        //std::cout << calledFuncName << " ";
                    }
                }
                //std::cout << std::endl;
            }
            //std::cout<<std::endl<<std::endl;
        }




        /**
         * @brief Update the callGraph map
         * @param func The specialized function
         * @return Nothing (could return error code?) 
         */
        void updateCallGraph(func::FuncOp func) {
            // Get the module containing this function
            auto module = func->getParentOfType<ModuleOp>();

            std::string funcName = func.getName().str();
            size_t pos = funcName.find('(');
            if (pos != std::string::npos) {
                funcName =  funcName.substr(0, pos);    
            }
            //std::cout << "FUNCNAME DEBUG: " << funcName << std::endl;
            // Initialize the entry for this function in the call graph if not already present
            if (callGraph.find(funcName) == callGraph.end()) {
                callGraph[funcName] = {};
            } else {
                // If it was initialized already, return immediately. Specialized functions always call the same!
                //std::cout << "RETURNING CAUSE " << funcName << " IS ALREADY INITIALIZED!" << std::endl;
                return;
            }
            func.walk([&](Operation *op) {
                // Check if the operation is a custom function call (e.g., "daphne.generic_call")
                if (op->getName().getStringRef() == "daphne.generic_call") {
                    if (auto calleeAttr = op->getAttrOfType<StringAttr>("callee")) {
                        std::string calleeName = calleeAttr.getValue().str();
                        
                        // Extract the input types (operand types)
                        TypeRange inputTypes = op->getOperandTypes();

                        // Extract the input values (operands)
                        ValueRange inputValues = op->getOperands();

                        // Use the operation's name as the base function name

                        // Generate the specialized function name
                        std::string specializedName = uniqueSpecializedFuncName(calleeName , inputTypes, inputValues);
                        size_t pos = specializedName.find('(');
                        if (pos != std::string::npos) {
                            specializedName =  specializedName.substr(0, pos);    
                        }
                        // Print the specialized function name
                        callGraph[funcName].insert(specializedName);
                    }
                }
            });
        }

        /**
         * @brief Create a specialized version of the template function.
         * @param templateFunction The template function.
         * @param specializedTypes The specialized function arguments
         * @param operands The operands of the call operation
         * @return The specialized function
         */
        func::FuncOp createSpecializedFunction(func::FuncOp templateFunction, TypeRange specializedTypes, ValueRange operands) {
            OpBuilder builder(templateFunction);
            auto specializedFunc = templateFunction.clone();
            builder.insert(specializedFunc);

            auto uniqueFuncName = uniqueSpecializedFuncName(templateFunction.getSymName().str(), specializedTypes, operands);
            specializedFunc.setName(uniqueFuncName);
            functions.insert({uniqueFuncName, specializedFunc});

            // change argument types
            specializedFunc
                .setType(builder.getFunctionType(specializedTypes, specializedFunc.getFunctionType().getResults()));
            for(auto it : llvm::zip(specializedFunc.getArguments(), specializedTypes)) {
                std::get<0>(it).setType(std::get<1>(it));
            }

            bool insertedConst = false;
            // Don't propagate constants into untyped functions, since that still causes problems for some reason.
            if(userConfig.use_ipa_const_propa && !isUntypedFunction(templateFunction)) {
                // Insert compile-time constant scalar call operands into the function.
                Block & specializedFuncBodyBlock = specializedFunc.getBody().front();
                builder.setInsertionPointToStart(&specializedFuncBodyBlock);
                for(auto it : llvm::enumerate(operands)) {
                    auto i = it.index();
                    Value v = it.value();
                    if(Operation * co = CompilerUtils::constantOfAnyType(v)) {
                        // Clone the constant operation into the function body.
                        Operation * coNew = co->clone();
                        builder.insert(coNew);
                        // Replace all uses of the corresponding block argument by the newly inserted constant.
                        specializedFuncBodyBlock.getArgument(i).replaceAllUsesWith(coNew->getResult(0));
                        // TODO We could even remove the corresponding function argument.
                        insertedConst = true;
                    }
                }
            }
            // Remember the newly specialized function for reuse only if we did not insert any constant
            // call operands.
            // TODO We could reuse it for other calls with the same constant (it's just more book-keeping effort).
            if(!insertedConst)
                specializedVersions.insert({templateFunction.getSymName().str(), specializedFunc});
            
            updateCallGraph(inferTypesInFunction(specializedFunc));

            findRecursions();
            for (const auto& cycle : recursiveCalls) {
                //std::cout << "Cycle detected: ";
                for (const auto& func : cycle) {
                    //std::cout << func << " ";
                }
                //std::cout << std::endl;
            }
            //std::cout << std::endl;
            printCallGraph();
            return inferTypesInFunction(specializedFunc);
        }

        /**
         * @brief Try to reuse an existing specialization for the given template function
         * @param specializedName The specialized name following the convention
         * @return either an existing and matching `FuncOp`, `nullptr` otherwise
         */
        func::FuncOp tryReuseExistingSpecialization(std::string specializedName) {
            auto it = functions.find(specializedName);
            if(it != functions.end()) {
                return it->second;
            }
            return nullptr;
        }

        /**
         * @brief Try to reuse an existing specializtion if one exists, else creates a new 
         *  specialization
         * @param operandTypes Operand types of the call operation
         * @param operands Operands of the call operation or an empty list if the operands are not available
         * @param calledFunction The function called by the call operation
         * @param callLoc The location of the call for which a function specialization shall be created or reused
         * @return A `FuncOp`for the specialization
         */
        func::FuncOp createOrReuseSpecialization(TypeRange operandTypes, ValueRange operands, func::FuncOp calledFunction, mlir::Location callLoc) {
            // check for existing specialization that matches
            auto specializedTypes = getSpecializedFuncArgTypes(calledFunction.getFunctionType(), operandTypes, calledFunction.getSymName().str(), callLoc); 
            auto specializedName = uniqueSpecializedFuncName(calledFunction.getSymName().str(), specializedTypes,operands );
            func::FuncOp specializedFunc = tryReuseExistingSpecialization(specializedName);
            if(!specializedFunc) {
                // Create specialized function
                specializedFunc = createSpecializedFunction(calledFunction, specializedTypes, operands);
            }
            if(logger->should_log(spdlog::level::debug)) {
                std::string s;
                llvm::raw_string_ostream stream(s);
                calledFunction->getLoc().print(stream);
                logger->debug("calledFunction\n\tname: {}\n\tlocation: {}", calledFunction.getSymName().str(), s);
            }
            templateFunctions.insert(calledFunction);
            return specializedFunc;
        }

        /**
         * @brief Recursively specializes all functions within a `FuncOp` based on calls to the functions
         * @param function The `FuncOp` to scan for function specializations
         */
        void specializeCallsInFunction(func::FuncOp function) {

            if(visited.count(function)) {
                return;
            }
            visited.insert(function);
            
            // Specialize all functions called directly
            function.walk([&](daphne::GenericCallOp callOp) {
                auto calledFunction = functions[callOp.getCallee().str()];
                bool hasConstantInput = llvm::any_of(
                        callOp.getOperands(),
                        [&](Value v) {
                            return CompilerUtils::constantOfAnyType(v) != nullptr;
                        }
                );

                if(isFunctionTemplate(calledFunction) || hasConstantInput) {
                    func::FuncOp specializedFunc = createOrReuseSpecialization(callOp.getOperandTypes(), callOp.getOperands(), calledFunction, callOp.getLoc());

                    // Check for duplicates
                    std::vector<std::string> specializedBody = getFunctionBody(specializedFunc);
                    for (auto &[originalName, originalFunc] : functions) {
                        if (areFunctionsSimilar(getFunctionBody(originalFunc), specializedBody)) {
                            // Mark as duplicate
                            duplicateFunctions[specializedFunc.getSymName().str()] = originalFunc.getSymName().str();

                            // Replace call to the duplicate with the original function
                            callOp.setCalleeAttr(originalFunc.getSymNameAttr());

                            /********Delete the duplicate specialized function from the IR **********/
                            specializedFunc.getOperation()->erase();
                            
                            // Return early as we found a duplicate
                            return;
                        }
                    }

                    callOp.setCalleeAttr(specializedFunc.getSymNameAttr());
                    if(fixResultTypes(callOp->getResults(), specializedFunc.getFunctionType())) {
                        inferTypesInFunction(function);
                    }
                    specializeCallsInFunction(specializedFunc);
                    called.insert(specializedFunc);
                }
                else {
                    specializeCallsInFunction(calledFunction);
                    called.insert(calledFunction);
                }
            });

            // Specialize all functions called by MapOp
            function.walk([&](daphne::MapOp mapOp) {
                auto calledFunction = functions[mapOp.getFunc().str()];
                if(isFunctionTemplate(calledFunction)) {
                     // Get the element type of the matrix the function should be mapped on
                    mlir::Type opTy = mapOp.getArg().getType();
                    auto inpMatrixTy = opTy.dyn_cast<daphne::MatrixType>();
                    func::FuncOp specializedFunc = createOrReuseSpecialization(inpMatrixTy.getElementType(), {}, calledFunction, mapOp.getLoc());
                    
                    // Check for duplicates
                    std::vector<std::string> specializedBody = getFunctionBody(specializedFunc);
                    for (auto &[originalName, originalFunc] : functions) {
                        if (areFunctionsSimilar(getFunctionBody(originalFunc), specializedBody)) {
                            
                            // Mark as duplicate
                            duplicateFunctions[specializedFunc.getSymName().str()] = originalFunc.getSymName().str();
                            
                            // Replace call to the duplicate with the original function
                            mapOp.setFuncAttr(originalFunc.getSymNameAttr());

                            /********Delete the duplicate specialized function from the IR **********/
                            specializedFunc.getOperation()->erase();

                            // Return early as we found a duplicate and deleted it
                            return;
                        }
                    }
                    
                    mapOp.setFuncAttr(specializedFunc.getSymNameAttr());

                    // We only allow functions that return exactly one result for mapOp
                    if (specializedFunc.getFunctionType().getNumResults() != 1) {
                        throw ErrorHandler::compilerError(
                            mapOp.getOperation(),
                            "SpecializeGenericFunctionsPass",
                            "map expects a function with exactly one return "
                            "value. The provided function returns" +
                                std::to_string(specializedFunc.getFunctionType()
                                                   .getNumResults()) +
                                "values instead.");
                    }

                    // Get current mapOp result matrix type and fix it if needed.
                    // If we fixed something we rerun inference of the whole function
                    daphne::MatrixType resMatrixTy = mapOp.getType().dyn_cast<daphne::MatrixType>();
                    mlir::Type funcResTy = specializedFunc.getFunctionType().getResult(0);

                    // The matrix that results from the mapOp has the same dimension as the input 
                    // matrix and the element-type returned by the specialized function
                    if(resMatrixTy.getNumCols() != inpMatrixTy.getNumCols() || 
                        resMatrixTy.getNumRows() != inpMatrixTy.getNumRows() ||
                        resMatrixTy.getElementType() != funcResTy) {
                        mapOp.getResult().setType(inpMatrixTy.withElementType(funcResTy));
                        inferTypesInFunction(function);
                    }

                    specializeCallsInFunction(specializedFunc);
                    called.insert(specializedFunc);
                }
                else {
                    specializeCallsInFunction(calledFunction);
                    called.insert(calledFunction);
                }
            });
        }

    public:
        void runOnOperation() final;

    StringRef getArgument() const final { return "specialize-generic-funcs"; }
    StringRef getDescription() const final { return "TODO"; }
    };
}

/**
 * @brief Generate and call specialized functions from template definitions and remove templates.
 *
 * We start entry functions (like `main` or `dist`) and then proceed as follows:
 *
 * 1. Infer types (types up to the first `GenericCallOp` will be inferred for sure)
 * 2. If the function called by `GenericCallOp` is untyped (input types are unknown), we clone it and set the input types
 *      to the types used in the call. For this specialized function we then do the same steps starting at 1.
 * 3. With the (possibly cloned) specialized function we now know the outputs. Starting here we infer up to the next
 *      `GenericCallOp` and go back to step 2.
 * 4. When all `GenericCallOp`s are specialized we are finished
 *
 * Finally we delete all the template functions such that the MLIR code can be verified for correct input and output types.
 */
void SpecializeGenericFunctionsPass::runOnOperation() {
    auto module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
        functions.insert({funcOp.getSymName().str(), funcOp});
    });

    // `entryFunctions` will hold entry functions like `main`, but also `dist` (for distributed computation)
    // we could also directly specify the names `main`, `dist` etc. (if we add more `entry` functions), or just set
    // an attribute flag for those functions.
    std::vector<func::FuncOp> entryFunctions;
    for(const auto &entry : functions) {
        entryFunctions.push_back(entry.second);
    }

    // Replace constants in each function before specialization
    for (auto &entry : functions) {
        replaceConstantsWithVariables(entry.second); // Replaces constants with variables
    }

    // Before specializing, check for duplicates using AST-based comparison
    checkForDuplicateSpecializationsAST(functions);

    for(const auto &function : entryFunctions) {
        if(isFunctionTemplate(function) || visited.count(function) || templateFunctions.count(function))
            continue;
        try {
            inferTypesInFunction(function);
        } catch (std::runtime_error& e) {
            throw ErrorHandler::rethrowError("SpecializeGenericFunctionsPass", e.what());
        }
        specializeCallsInFunction(function);
    }

    // Delete non-called functions.
    for(auto f : functions) {
        // Never remove the main or dist function.
        if(f.first == "main" or f.first == "dist")
            continue;
        // Remove a function that was present before creating specializations,
        // if it is never called.
        if(!called.count(f.second) || templateFunctions.count(f.second))
            f.second.erase();
    }
}

std::unique_ptr<Pass> daphne::createSpecializeGenericFunctionsPass(const DaphneUserConfig& cfg) {
    
    return std::make_unique<SpecializeGenericFunctionsPass>(cfg);
}
